import math 
import logging 
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Union
import torch 
import torch.nn as nn 
from torch import Tensor 
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import SAGEConv, GCNConv, GATConv
import faiss
import re

logger = logging.getLogger(__name__)

def generate_relative_position_matrix(length, max_relative_position, use_negative_distance):
    """
    Generate the clipped relative position matrix.
    """
    range_vec = torch.arange(length)
    range_matrix = range_vec.unsqueeze(1).expand(-1, length).transpose(0,1)
    distance_matrix = range_matrix - range_matrix.transpose(0, 1)
    distance_mat_clipped = torch.clamp(distance_matrix, min=-max_relative_position, max=max_relative_position)

    if use_negative_distance:
        final_matrix = distance_mat_clipped + max_relative_position
    else:
        final_matrix = torch.abs(distance_mat_clipped)

    return final_matrix

def freeze_params(module: nn.Module) -> None:
    """
    freeze the parameters of this module,
    i.e. do not update them during training
    """
    for _, p in module.named_parameters():
        p.requires_grad = False

def subsequent_mask(size:int) -> Tensor:
    """
    Mask out subsequent positions (to prevent attending to future position)
    size: trg_len
    return:
        [1, size, size] (bool values)
    """
    ones = torch.ones(size,size, dtype=torch.bool)
    return torch.tril(ones, out=ones).unsqueeze(0) 

class XentLoss(nn.Module):
    """
    Cross-Entropy loss with optional label smoothing.
    reduction='sum' means add all sequences and all tokens loss in the batch.
    reduction='mean' means take average of all sequence and all token loss in the batch.
    """
    def __init__(self, pad_index: int, smoothing: float=0.0) -> None:
        super().__init__()
        self.smoothing = smoothing
        self.pad_index = pad_index 
        if self.smoothing <= 0.0:
            self.criterion = nn.NLLLoss(ignore_index=self.pad_index, reduction="sum")
        else:
            self.criterion = nn.KLDivLoss(reduction="sum")

    def reshape(self, log_probs: Tensor, target: Tensor) -> Tensor:
        """
        Reshape Tensor because of the input restrict of nn.NLLLoss/nn.CrossEntropyLoss
        :param log_probs [batch_size, trg_len, vocab_size]
        :param target [batch_size, trg_len]
        """
        vocab_size = log_probs.size(-1)
        log_probs = log_probs.contiguous().view(-1, vocab_size)
        # log_probs [batch_size*trg_len, vocab_size]

        if self.smoothing > 0:
            target = self.smooth_target(target.contiguous().view(-1), vocab_size)
        else:
            target = target.contiguous().view(-1)
            # target [batch_size*trg_len]

        return log_probs, target

    def smooth_target(self, target, vocab_size):
        """
        target: [batch_size*trg_len]
        vocab_size: a number
        return: smoothed target distributions, batch*trg_len x vocab_size
        """
        # batch*trg_len x vocab_size
        smooth_dist = target.new_zeros((target.size(0), vocab_size)).float()
        # fill distribution uniformly with smoothing
        smooth_dist.fill_(self.smoothing / (vocab_size - 2))
        # assign true label the probability of 1-smoothing ("confidence")
        smooth_dist.scatter_(1, target.unsqueeze(1).data, 1.0 - self.smoothing)
        # give padding probability of 0 everywhere
        smooth_dist[:, self.pad_index] = 0
        # masking out padding area (sum of probabilities for padding area = 0)
        padding_positions = torch.nonzero(target.data == self.pad_index,
                                          as_tuple=False)
        if len(padding_positions) > 0:
            smooth_dist.index_fill_(0, padding_positions.squeeze(), 0.0)
        return Variable(smooth_dist, requires_grad=False)

    def forward(self, log_probs: Tensor, target: Tensor) -> Tensor:
        """
        Compute the cross-entropy between logits and targets.
        :param log_probs [batch_size, trg_len, vocab_size]
        :param target [batch_size, trg_len]
        """
        log_probs, target = self.reshape(log_probs, target)
        batch_loss = self.criterion(log_probs, target)
        return batch_loss
    
    # def __repr__(self):
    #     return (f"{self.__class__.__name__}(criterion={self.criterion}, "
    #             f"smoothing={self.smoothing})")

class Embeddings(nn.Module):
    def __init__(self, embedding_dim:int=64,
                 scale: bool=True, vocab_size:int=0,
                 padding_index:int=1, freeze:bool=False) -> None:
        """
        scale: for transformer see "https://zhuanlan.zhihu.com/p/442509602"
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.scale = scale
        self.vocab_size = vocab_size
        self.lut = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=padding_index)
        if freeze:
            freeze_params(self)

    def forward(self, source: Tensor) -> Tensor:
        """
        Perform lookup for input(source) in the embedding table.
        return the embedded representation for source.
        """
        if self.scale:
            return self.lut(source) * math.sqrt(self.embedding_dim)
        else:
            return self.lut(source)

    # def __repr__(self):
    #     return (f"{self.__class__.__name__}("
    #             f"embedding_dim={self.embedding_dim}, "
    #             f"vocab_size={self.vocab_size})")
    
    def load_from_file(self, embed_path:Path, vocab=None) -> None:
        """
        Load pretrained embedding weights from text file.
        - First line is expeceted to contain vocabulary size and dimension.
        The dimension has to match the model's specified embedding size, the vocabulary size is used in logging only.
        - Each line should contain word and embedding weights separated by spaces.
        - The pretrained vocabulary items that are not part of the vocabulary will be ignored (not loaded from the file).
        - The initialization of Vocabulary items that are not part of the pretrained vocabulary will be kept
        - This function should be called after initialization!
        Examples:
            2 5
            the -0.0230 -0.0264  0.0287  0.0171  0.1403
            at -0.0395 -0.1286  0.0275  0.0254 -0.0932        
        """
        embed_dict: Dict[int,Tensor] = {}
        with embed_path.open("r", encoding="utf-8",errors="ignore") as f_embed:
            vocab_size, dimension = map(int, f_embed.readline().split())
            assert self.embedding_dim == dimension, "Embedding dimension doesn't match."
            for line in f_embed.readlines():
                tokens = line.rstrip().split(" ")
                if tokens[0] in vocab.specials or not vocab.is_unk(tokens[0]):
                    embed_dict[vocab.lookup(tokens[0])] = torch.FloatTensor([float(t) for t in tokens[1:]])
            
            logging.info("Loaded %d of %d (%%) tokens in the pre-trained file.",
            len(embed_dict), vocab_size, len(embed_dict)/vocab_size)

            for idx, weights in embed_dict.items():
                if idx < self.vocab_size:
                    assert self.embedding_dim == len(weights)
                    self.lut.weight.data[idx] == weights
            
            logging.info("Cover %d of %d (%%) tokens in the Original Vocabulary.",
            len(embed_dict), len(vocab), len(embed_dict)/len(vocab))

class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from 'Attention is all you need'.
    consider relative position.
    """
    def __init__(self, head_count: int, model_dim:int, dropout: float=0.1,
                 max_relative_position=0, use_negative_distance=False) -> None:
        super().__init__()
        assert model_dim % head_count == 0, 'model dim must be divisible by head count'

        self.head_size = model_dim // head_count
        self.head_count = head_count
        self.model_dim = model_dim

        self.key_project = nn.Linear(model_dim, head_count * self.head_size)
        self.query_project = nn.Linear(model_dim, head_count * self.head_size)
        self.value_project = nn.Linear(model_dim, head_count * self.head_size)

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.output_layer = nn.Linear(model_dim, model_dim)

        self.max_relative_position = max_relative_position
        self.use_negative_distance = use_negative_distance

        if self.max_relative_position > 0:
            relative_position_size = self.max_relative_position*2+1 if self.use_negative_distance is True else self.max_relative_position+1
            self.relative_position_embedding_key = nn.Embedding(relative_position_size, self.head_size)
            self.relative_position_embedding_value = nn.Embedding(relative_position_size, self.head_size)

    def forward(self, key, value, query, mask=None):
        """
        Compute multi-headed attention.
        key  [batch_size, seq_len, hidden_size]
        value[batch_size, seq_len, hidden_size]
        query[batch_size, seq_len, hidden_size]
        mask [batch_size, 1 or seq_len, seq_len] (pad position is false or zero)

        return 
            - output [batch_size, query_len, hidden_size]
            - attention_output_weights [batch_size, query_len, key_len]
        """
        batch_size = key.size(0)
        key_len = key.size(1)
        query_len = query.size(1)
        value_len = value.size(1)

        # project query key value
        key = self.key_project(key)
        value = self.value_project(value)
        query = self.query_project(query)

        #reshape key, value, query 
        key = key.view(batch_size, -1, self.head_count, self.head_size).transpose(1,2)
        #[batch_size, head_count, key_len, head_size]
        value = value.view(batch_size, -1, self.head_count, self.head_size).transpose(1,2)
        query = query.view(batch_size, -1, self.head_count, self.head_size).transpose(1,2)

        # scale and calculate attention scores
        query = query / math.sqrt(self.head_size)
        scores = torch.matmul(query, key.transpose(2,3))
        # scores [batch_size, head_count, query_len, key_len]

        if self.max_relative_position > 0: 
            relative_position_matrix = generate_relative_position_matrix(key_len, self.max_relative_position, self.use_negative_distance)
            relative_position_matrix = relative_position_matrix.to(key.device)
            relative_key = self.relative_position_embedding_key(relative_position_matrix)
            # relative_key [key_len, key_len, head_size]
            relative_vaule = self.relative_position_embedding_value(relative_position_matrix)
            # relative_value [value_len, value_len, head_size]
            r_query = query.permute(2,0,1,3).reshape(query_len, batch_size*self.head_count, self.head_size)
            assert query_len == key_len, "For relative position."
            scores_relative = torch.matmul(r_query, relative_key.transpose(1,2)).reshape(query_len, batch_size, self.head_count, key_len)
            scores_relative = scores_relative.permute(1, 2, 0, 3)
            scores = scores + scores_relative

        # apply mask Note: add a dimension to mask -> [batch_size, 1, 1 or len , key_len]
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        
        # apply attention dropout
        attention_weights = self.softmax(scores) # attention_weights [batch_size, head_count, query_len, key_len]
        attention_probs = self.dropout(attention_weights)

        # get context vector
        context = torch.matmul(attention_probs, value) # context [batch_size, head_count, query_len, head_size]
        context = context.transpose(1,2).contiguous().view(batch_size, -1, self.head_count*self.head_size)
        # context [batch_size, query_len, hidden_size]

        if self.max_relative_position > 0:
            r_attention_probs = attention_probs.permute(2,0,1,3).reshape(query_len, batch_size*self.head_count, key_len)
            context_relative = torch.matmul(r_attention_probs, relative_vaule) # context_relative [query_len, batch_size*self.head_count, head_size]
            context_relative = context_relative.reshape(query_len, batch_size, self.head_count, self.head_size).permute(1, 2, 0, 3)
            context_relative = context_relative.transpose(1, 2).contiguous().view(batch_size, -1, self.head_count*self.head_size)
            # context_relative [batch_size, query_len, hidden_size]
            context = context + context_relative

        output = self.output_layer(context)

        attention_output_weights = attention_weights.view(batch_size, self.head_count, query_len, key_len).sum(dim=1) / self.head_count

        return output, attention_output_weights

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-forward layer (FF)
    Projects to ff_size and then back dowm to input_dim.
    Pre-LN and Post-LN Transformer cite "Understanding the Difficulity of Training Transformers"
    """
    def __init__(self, model_dim:int, ff_dim:int, dropout:float=0.1, layer_norm_position:str="post") -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(model_dim)
        self.pwff = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim),
            nn.Dropout(dropout),
        )
        self.layer_norm_position = layer_norm_position
        assert self.layer_norm_position in {"pre","post"}
    
    def forward(self, x:Tensor) -> Tensor:
        """
        Layer definition.
        input x: [batch_size, seq_len, model_dim]
        output: [batch_size, seq_len, model_dim]
        """
        residual = x
        if self.layer_norm_position == "pre":
            x = self.layer_norm(x)
        x = self.pwff(x) + residual
        if self.layer_norm_position == "post":
            x = self.layer_norm(x)
        return x

class LearnablePositionalEncoding(nn.Module):
    """
    Learnable position encodings. (used in Bert etc.)
    """
    def __init__(self, model_dim:int=0, max_len:int=512) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.max_len = max_len
        self.learn_lut = nn.Embedding(max_len, self.model_dim)

    def forward(self, embed: Tensor) -> Tensor:
        """
        Perform lookup for input(source) in the learnable embeding position table.
        :param src_input [batch_size, src_len]
        :param embed_src [batch_size, src_len, embed_dim]
        return embed_src + lpe(src_input)
        """
        batch_size = embed.size(0)
        len = embed.size(1)
        assert len <= self.max_len, 'length must <= max len'
        position_input = torch.arange(len).unsqueeze(0).repeat(batch_size, 1).to(embed.device)
        # make sure embed and position embed have same scale.
        return embed + self.learn_lut(position_input)

class TransformerEncoderLayer(nn.Module):
    """
    Classical transformer Encoder layer
    containing a Multi-Head attention layer and a position-wise feed-forward layer.
    """
    def __init__(self, model_dim:int, ff_dim:int, head_count:int, 
                 dropout:float=0.1, layer_norm_position:str="pre",
                 max_relative_position:int=0, use_negative_distance:bool=False) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(model_dim)
        self.src_src_attenion = MultiHeadedAttention(head_count, model_dim, dropout, max_relative_position, use_negative_distance)
        self.feed_forward = PositionwiseFeedForward(model_dim, ff_dim, dropout=dropout, layer_norm_position=layer_norm_position)
        self.dropout = nn.Dropout(dropout)

        self.layer_norm_position = layer_norm_position
        assert self.layer_norm_position in {'pre','post'}
    
    def forward(self, input:Tensor, mask:Tensor) -> Tensor:
        """
        Forward pass for a single transformer encoder layer.
        input [batch_size, src_len, model_dim]
        mask [batch_size, 1, src_len]
        return:
            output [batch_size, src_len, model_dim]
        """
        residual = input
        if self.layer_norm_position == "pre":
            input = self.layer_norm(input)
        attention_output, _ = self.src_src_attenion(input, input, input, mask)
        feedforward_input = self.dropout(attention_output) + residual

        if self.layer_norm_position == "post":
            feedforward_input = self.layer_norm(feedforward_input)

        output = self.feed_forward(feedforward_input)
        return output

class GNNEncoderLayer(nn.Module):
    """
    Classical GNN model encoder layer.
    """
    def __init__(self, model_dim=512, GNN=None, aggr=None) -> None:
        super().__init__()

        assert GNN in {"SAGEConv", "GCNConv", "GATConv"}
        self.gnn = None 
        if GNN == "SAGEConv":
            self.gnn = SAGEConv(in_channels=model_dim, out_channels=model_dim, aggr=aggr)
        elif GNN == "GCNConv":
            self.gnn = GCNConv(in_channels=model_dim, out_channels=model_dim, aggr=aggr)
        elif GNN == "GATConv":
            self.gnn = GATConv(in_channels=model_dim, out_channels=model_dim, aggr=aggr)
        
        self.relu = nn.ReLU()
        # self.dropout= nn.Dropout(0.2)
        self.layer_norm = nn.LayerNorm(model_dim)
    
    def forward(self, node_feature, edge_index):
        """
        Input:
            node_emb [batch, node_emb_dim] / node feature tensor
            edge_index: Adj [2, edge_num] 
        Return: 
            node_encode [batch, node_num, node_dim]
        """
        residual = node_feature 
        node_feature = self.layer_norm(node_feature)
        node_enc_ = self.gnn(x=node_feature, edge_index=edge_index)
        node_enc_ = self.relu(node_enc_)
        # node_encode = node_enc_ + residual
        return node_enc_

class TransformerDecoderLayer(nn.Module):
    """
    Classical transformer Decoder Layer
    """
    def __init__(self, model_dim:int, ff_dim:int,head_count:int,
                 dropout:float=0.1, layer_norm_position:str='pre',
                 max_relative_position:int=0, use_negative_distance:bool=False) -> None:
        "layer norm position either 'pre' or 'post'."
        super().__init__()
        self.model_dim = model_dim
        self.trg_trg_attention = MultiHeadedAttention(head_count, model_dim, dropout, max_relative_position, use_negative_distance)
        self.src_trg_attention = MultiHeadedAttention(head_count, model_dim, dropout, max_relative_position=0, use_negative_distance=False)
        self.gnn_trg_attention = MultiHeadedAttention(head_count, model_dim, dropout, max_relative_position=0, use_negative_distance=False)

        self.feed_forward = PositionwiseFeedForward(model_dim, ff_dim, dropout=dropout, layer_norm_position=layer_norm_position)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.layer_norm2 = nn.LayerNorm(model_dim)
        self.layer_norm3 = nn.LayerNorm(model_dim)

        self.layer_norm_position = layer_norm_position
        assert self.layer_norm_position in {'pre','post'}

    def forward(self, trg_input:Tensor, code_encoder_memory:Tensor, ast_encoder_memory:Tensor,
                src_mask: Tensor, node_mask:None, trg_mask:Tensor) -> Tensor:
        """
        Forward pass for a single transformer decoer layer.
        input [batch_size, trg_len, model_dim]
        memory [batch_size, src_len, model_dim]
        src_mask [batch_size, 1, src_len]
        trg_mask [batch_size, trg_len, trg_len]
        return:
            output [batch_size, trg_len, model_dim]
            cross_attention_weight [batch_size, trg_len, src_len]
        """
        residual = trg_input
        if self.layer_norm_position == 'pre':
            trg_input = self.layer_norm(trg_input)
        self_attention_output, _ = self.trg_trg_attention(trg_input,trg_input,trg_input, mask=trg_mask)
        cross_attention_input = self.dropout(self_attention_output) + residual

        if self.layer_norm_position == 'post':
            cross_attention_input = self.layer_norm(cross_attention_input)
        
        cross_residual = cross_attention_input
        if self.layer_norm_position == "pre":
            cross_attention_input = self.layer_norm2(cross_attention_input)
        cross_attention_output, cross_attention_weight = self.gnn_trg_attention(ast_encoder_memory, ast_encoder_memory,
                                                                                cross_attention_input, mask=node_mask)
        src_trg_attention_input = self.dropout(cross_attention_output) + cross_residual
        if self.layer_norm_position == 'post':
            src_trg_attention_input = self.layer_norm2(src_trg_attention_input)

        cross_residual = src_trg_attention_input
        if self.layer_norm_position == 'pre':
            src_trg_attention_input = self.layer_norm3(src_trg_attention_input)
        cross_attention_output, cross_attention_weight = self.src_trg_attention(code_encoder_memory, code_encoder_memory, src_trg_attention_input,mask=src_mask)
        feedforward_input = self.dropout(cross_attention_output) + cross_residual

        if self.layer_norm_position == 'post':
            feedforward_input = self.layer_norm3(feedforward_input)

        output = self.feed_forward(feedforward_input)
        return output, cross_attention_weight
    
    def context_representation(self, penultimate:Tensor, code_encoder_memory:Tensor, src_mask:Tensor,
                               ast_encoder_memory:Tensor, node_mask:Tensor, trg_mask:Tensor) -> Tensor:
        """
        Get the hidden state for search purpose.
        The hidden state means the token semantic.
        """
        residual = penultimate
        if self.layer_norm_position == 'pre':
            penultimate = self.layer_norm(penultimate)
        self_attention_output, _ = self.trg_trg_attention(penultimate,penultimate,penultimate, mask=trg_mask)
        cross_attention_input = self.dropout(self_attention_output) + residual

        if self.layer_norm_position == 'post':
            cross_attention_input = self.layer_norm(cross_attention_input)
        
        cross_residual = cross_attention_input
        if self.layer_norm_position == "pre":
            cross_attention_input = self.layer_norm2(cross_attention_input)
        cross_attention_output, cross_attention_weight = self.gnn_trg_attention(ast_encoder_memory, ast_encoder_memory,
                                                                                cross_attention_input, mask=node_mask)
        src_trg_attention_input = self.dropout(cross_attention_output) + cross_residual
        if self.layer_norm_position == 'post':
            src_trg_attention_input = self.layer_norm2(src_trg_attention_input)

        cross_residual = src_trg_attention_input
        if self.layer_norm_position == 'pre':
            src_trg_attention_input = self.layer_norm3(src_trg_attention_input)
        cross_attention_output, cross_attention_weight = self.src_trg_attention(code_encoder_memory, code_encoder_memory, src_trg_attention_input,mask=src_mask)
        feedforward_input = self.dropout(cross_attention_output) + cross_residual

        if self.layer_norm_position == 'post':
            feedforward_input = self.layer_norm3(feedforward_input)

        representation = self.feed_forward.layer_norm(feedforward_input)
        return representation

class TransformerEncoder(nn.Module):
    """
    Classical Transformer Encoder.
    """
    def __init__(self, model_dim:int=512, ff_dim:int=2048, 
                 num_layers:int=6, head_count:int=8, dropout:float=0.1, 
                 emb_dropout:float=0.1, layer_norm_position:str='pre', 
                 src_pos_emb:str="absolute", max_src_len:int=512, freeze:bool=False,
                 max_relative_position:int=0, use_negative_distance:bool=False) -> None:
        super().__init__()

        self.layers = nn.ModuleList([TransformerEncoderLayer(model_dim, ff_dim, head_count, dropout,
                    layer_norm_position, max_relative_position, use_negative_distance) for _ in range(num_layers)])
        
        self.layer_norm = nn.LayerNorm(model_dim) if layer_norm_position == 'pre' else None
        self.head_count = head_count
        self.layer_norm_position = layer_norm_position
        if freeze:
            freeze_params(self)
    
    def forward(self, embed_src:Tensor, mask:Tensor=None) -> Tensor:
        """
        Pass the input and mask through each layer in turn.
        embed_src [batch_size, src_len, embed_size]
        mask: indicates padding areas (zeros where padding) [batch_size, 1, src_len]
        """
        input = embed_src
        for layer in self.layers:
            input = layer(input, mask)
        
        if self.layer_norm is not None: # for Pre-LN Transformer
            input = self.layer_norm(input) 
        
        return input
    
    # def __repr__(self):
    #     return (f"{self.__class__.__name__}(num_layers={len(self.layers)}, "
    #             f"head_count={self.head_count}, " 
    #             f"layer_norm_position={self.layer_norm_position})")

class GNNEncoder(nn.Module):
    def __init__(self, gnn_type, aggr, model_dim, num_layers) -> None:
        super().__init__()
        self.layers = nn.ModuleList([GNNEncoderLayer(model_dim=model_dim, GNN=gnn_type, aggr=aggr)
                                      for _ in range(num_layers)])
        
        self.layernorm = nn.LayerNorm(model_dim)
    
    def forward(self, node_feature, edge_index, node_batch):
        """
        Input: 
            node_feature: [node_number, node_dim]
            edge_index: [2, edge_number]
            node_batch: {0,0, 1, ..., B-1} | indicate node in which graph. 
                        B: the batch_size or graphs.
        Return 
            output: [batch, Nmax, node_dim]
            mask: [batch, Nmax] bool
            Nmax: max node number in a batch.
        """
        for layer in self.layers:
            node_feature = layer(node_feature, edge_index)
        
        node_feature = self.layernorm(node_feature)

        if node_batch is not None:
            output, mask = to_dense_batch(node_feature, batch=node_batch)

        return output, mask

class TransformerDecoder(nn.Module):
    """
    Classical Transformer Decoder
    """
    def __init__(self, model_dim:int=512, ff_dim:int=2048,
                 num_layers:int=6, head_count:int=8, dropout:float=0.1,
                 emb_dropout:float=0.1, layer_norm_position:str='pre',
                 trg_pos_emb:str="absolute", max_trg_len:int=512, freeze:bool=False,
                 max_relative_positon:int=0, use_negative_distance:bool=False) -> None:
        super().__init__()

        self.layers = nn.ModuleList([TransformerDecoderLayer(model_dim,ff_dim,head_count, dropout,
            layer_norm_position, max_relative_positon, use_negative_distance) for _ in range(num_layers)])

        self.layer_norm = nn.LayerNorm(model_dim) if layer_norm_position == 'pre' else None
        
        self.head_count = head_count
        self.layer_norm_position = layer_norm_position
        self.model_dim = model_dim
        if freeze:
            freeze_params(self)
    
    def forward(self, 
                transformer_encoder_output: Tensor,
                gnn_encoder_output:Tensor, 
                trg_input:Tensor,
                src_mask:Tensor,
                node_mask: Tensor, 
                trg_mask:Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Transformer decoder forward pass.
        embed_trg [batch_size, trg_len, model_dim]
        encoder_ouput [batch_size, src_len, model_dim]
        src_mask [batch_size, 1, src_len]
        node_mask [batch_size, node_len]
        trg_mask [batch_size, 1, trg_len]
        return:
            output [batch_size, trg_len, model_dim]  
            cross_attention_weight [batch_size, trg_len, src_len]
        """

        trg_mask = trg_mask & subsequent_mask(trg_input.size(1)).type_as(trg_mask)
        # trg_mask [batch_size, 1, trg_len] -> [batch_size, trg_len, trg_len] (include mask the token unseen)
        node_mask = node_mask.unsqueeze(1) # [batch_size, 1, node_len]

        penultimate = None
        for layer in self.layers:
            penultimate = trg_input
            trg_input, cross_attention_weight = layer(trg_input, code_encoder_memory=transformer_encoder_output, 
                ast_encoder_memory=gnn_encoder_output, src_mask=src_mask, node_mask=node_mask, trg_mask=trg_mask)
        
        penultimate_representation = self.layers[-1].context_representation(penultimate, transformer_encoder_output,
                                                        src_mask, gnn_encoder_output, node_mask, trg_mask)

        if self.layer_norm is not None:
            trg_input = self.layer_norm(trg_input)
        
        output = trg_input
        return output, penultimate_representation, cross_attention_weight
    
    # def __repr__(self):
    #     return (f"{self.__class__.__name__}(num_layers={len(self.layers)}, "
    #             f"head_count={self.head_count}, " 
    #             f"layer_norm_position={self.layer_norm_position})")

class Model(nn.Module):
    """
    Two encoder (transformer encoder + gnn encoder)
    One decoder (transformer decoder)
    """
    def __init__(self, model_cfg:dict=None, vocab_info:dict=None) -> None:
        super().__init__()
        self.vocab_info = vocab_info
        transformer_encoder_cfg = model_cfg["transformer_encoder"]
        self.transformer_encoder = TransformerEncoder(model_dim=transformer_encoder_cfg["model_dim"], 
                            ff_dim=transformer_encoder_cfg["ff_dim"],
                            num_layers=transformer_encoder_cfg["num_layers"],
                            head_count=transformer_encoder_cfg["head_count"],
                            dropout=transformer_encoder_cfg["dropout"],
                            layer_norm_position=transformer_encoder_cfg["layer_norm_position"],
                            src_pos_emb=transformer_encoder_cfg["src_pos_emb"],
                            max_src_len=transformer_encoder_cfg["max_src_len"],
                            freeze=transformer_encoder_cfg["freeze"],
                            max_relative_position=transformer_encoder_cfg["max_relative_position"],
                            use_negative_distance=transformer_encoder_cfg["use_negative_distance"])

        gnn_encoder_cfg = model_cfg["gnn_encoder"]
        self.gnn_encoder = GNNEncoder(gnn_type=gnn_encoder_cfg["gnn_type"],
                                      aggr=gnn_encoder_cfg["aggr"],
                                      model_dim=gnn_encoder_cfg["model_dim"],
                                      num_layers=gnn_encoder_cfg["num_layers"])

        transformer_decoder_cfg = model_cfg["transformer_decoder"]
        self.transformer_decoder = TransformerDecoder(model_dim=transformer_decoder_cfg["model_dim"],
                                ff_dim=transformer_decoder_cfg["ff_dim"],
                                num_layers=transformer_decoder_cfg["num_layers"],
                                head_count=transformer_decoder_cfg["head_count"],
                                dropout=transformer_decoder_cfg["dropout"], 
                                layer_norm_position=transformer_decoder_cfg["layer_norm_position"],
                                trg_pos_emb=transformer_decoder_cfg["trg_pos_emb"],
                                max_trg_len=transformer_decoder_cfg["max_trg_len"],
                                freeze=transformer_decoder_cfg["freeze"],
                                max_relative_positon=transformer_decoder_cfg["max_relative_position"],
                                use_negative_distance=transformer_decoder_cfg["use_negative_distance"])

        embedding_cfg = model_cfg["embeddings"]
        # src_embed: for code token and ast token embedding
        self.src_embed = Embeddings(embedding_dim=embedding_cfg['embedding_dim'],
                           scale=embedding_cfg['scale'],
                           vocab_size=vocab_info["src_vocab"]["size"],
                           padding_index=vocab_info["src_vocab"]["pad_index"],
                           freeze=embedding_cfg['freeze'])
        
        # learnable_embed: for code token with learnable position embedding 
        self.code_learnable_embed = LearnablePositionalEncoding(model_dim=transformer_encoder_cfg["model_dim"], 
                                    max_len=transformer_encoder_cfg["max_src_len"])

        # position_embed: for ast token with triplet position embedding 
        self.position_embed = Embeddings(embedding_dim=embedding_cfg['embedding_dim'],
                           scale=embedding_cfg['scale'],
                           vocab_size=vocab_info["position_vocab"]["size"],
                           padding_index=vocab_info["position_vocab"]["pad_index"],
                           freeze=embedding_cfg['freeze'])        
        
        # trg_embed: for text token embedding 
        self.trg_embed = Embeddings(embedding_dim=embedding_cfg['embedding_dim'],
                           scale=embedding_cfg['scale'],
                           vocab_size=vocab_info["trg_vocab"]["size"],
                           padding_index=vocab_info["trg_vocab"]["pad_index"],
                           freeze=embedding_cfg['freeze'])
        
        self.trg_learnable_embed = LearnablePositionalEncoding(model_dim=transformer_decoder_cfg["model_dim"],
                                                               max_len=transformer_decoder_cfg["max_trg_len"])
        
        self.code_emb_dropout = nn.Dropout(transformer_encoder_cfg["dropout"])
        self.ast_node_emb_dropout = nn.Dropout(gnn_encoder_cfg["dropout"])
        self.text_emb_dropout = nn.Dropout(transformer_decoder_cfg["dropout"])
        
        self.output_layer = nn.Linear(transformer_decoder_cfg["model_dim"], vocab_info["trg_vocab"]["size"], bias=False)

        self.loss_function = XentLoss(pad_index=vocab_info["trg_vocab"]["pad_index"], smoothing=0)

    def forward(self, return_type:str=None,
                src_input_code_token:Tensor=None, 
                src_input_ast_token:Tensor=None,
                src_input_ast_position:Tensor=None,
                node_batch: Tensor=None,
                edge_index: Tensor=None,
                trg_input:Tensor=None,
                trg_truth:Tensor=None,
                src_mask:Tensor=None,
                node_mask:Tensor=None,
                trg_mask:Tensor=None,
                transformer_encoder_output:Tensor=None,
                gnn_encoder_output:Tensor=None):
        """
        Input:
            return_type: loss, encode_decode, encode.
            src_input_code_token: [batch_size, src_code_token_len]
            src_input_ast_token: [src_ast_token_len(all batch)]
            src_input_ast_position: [src_ast_token_len(all batch)]
            node_batch: [src_ast_token_len(all batch)]
            edge_index: [2, edge_number(all batch)]
            trg_input: [batch_size, trg_len]
            trg_truth: [batch_size, trg_len]
            src_mask: [batch_size, 1, src_len] 0 means be ignored.
            trg_mask: [batch_size, 1, trg_len] 0 means be ignored.
        """
        if return_type == "loss":
            embed_src_code_token = self.src_embed(src_input_code_token)
            transformer_encoder_input = self.code_learnable_embed(embed_src_code_token)
            transformer_encoder_input = self.code_emb_dropout(transformer_encoder_input)

            embed_src_ast_token = self.src_embed(src_input_ast_token)
            gnn_encoder_input = self.position_embed(src_input_ast_position) + embed_src_ast_token
            gnn_encoder_input = self.ast_node_emb_dropout(gnn_encoder_input)

            transformer_encoder_output = self.transformer_encoder(transformer_encoder_input, src_mask)
            gnn_encoder_output, node_mask = self.gnn_encoder(gnn_encoder_input, edge_index, node_batch)

            embed_trg_input = self.trg_embed(trg_input)
            decoder_trg_input = self.trg_learnable_embed(embed_trg_input)
            decoder_trg_input = self.text_emb_dropout(decoder_trg_input)
            transformer_decoder_output, _, _ = self.transformer_decoder(transformer_encoder_output, 
                                        gnn_encoder_output, decoder_trg_input, src_mask, node_mask, trg_mask)
            
            logits = self.output_layer(transformer_decoder_output) 

            log_probs = F.log_softmax(logits, dim=-1)

            batch_loss = self.loss_function(log_probs, target=trg_truth)
            # NOTE batch loss = sum over all sentence of all tokens in the batch that are not pad!
            return batch_loss

        elif return_type == "encode":
            embed_src_code_token = self.src_embed(src_input_code_token)
            transformer_encoder_input = self.code_learnable_embed(embed_src_code_token)
            transformer_encoder_input = self.code_emb_dropout(transformer_encoder_input)

            embed_src_ast_token = self.src_embed(src_input_ast_token)
            gnn_encoder_input = self.position_embed(src_input_ast_position) + embed_src_ast_token
            gnn_encoder_input = self.ast_node_emb_dropout(gnn_encoder_input)

            transformer_encoder_output = self.transformer_encoder(transformer_encoder_input, src_mask)
            gnn_encoder_output, node_mask = self.gnn_encoder(gnn_encoder_input, edge_index, node_batch)
            return transformer_encoder_output, src_mask, gnn_encoder_output, node_mask
        
        elif return_type == "decode":
            embed_trg_input = self.trg_embed(trg_input)
            decoder_trg_input = self.trg_learnable_embed(embed_trg_input)
            decoder_trg_input = self.text_emb_dropout(decoder_trg_input)
            transformer_decoder_output, penultimate_representation, cross_attention_weight = self.transformer_decoder(transformer_encoder_output, 
                        gnn_encoder_output, decoder_trg_input, src_mask, node_mask, trg_mask)
            
            logits = self.output_layer(transformer_decoder_output)
            return logits, penultimate_representation, cross_attention_weight
        
        elif return_type == "get_penultimate_representation":
            embed_src_code_token = self.src_embed(src_input_code_token)
            transformer_encoder_input = self.code_learnable_embed(embed_src_code_token)
            transformer_encoder_input = self.code_emb_dropout(transformer_encoder_input)

            embed_src_ast_token = self.src_embed(src_input_ast_token)
            gnn_encoder_input = self.position_embed(src_input_ast_position) + embed_src_ast_token
            gnn_encoder_input = self.ast_node_emb_dropout(gnn_encoder_input)

            transformer_encoder_output = self.transformer_encoder(transformer_encoder_input, src_mask)
            gnn_encoder_output, node_mask = self.gnn_encoder(gnn_encoder_input, edge_index, node_batch)

            embed_trg_input = self.trg_embed(trg_input)
            decoder_trg_input = self.trg_learnable_embed(embed_trg_input)
            decoder_trg_input = self.text_emb_dropout(decoder_trg_input)
            transformer_decoder_output, penultimate_representation, cross_attention_weight = self.transformer_decoder(
                transformer_encoder_output, gnn_encoder_output, decoder_trg_input, src_mask, node_mask, trg_mask)
            
            return penultimate_representation
        
        elif return_type == "retrieval_adaptor":

            embed_src_code_token = self.src_embed(src_input_code_token)
            transformer_encoder_input = self.code_learnable_embed(embed_src_code_token)
            transformer_encoder_input = self.code_emb_dropout(transformer_encoder_input)

            embed_src_ast_token = self.src_embed(src_input_ast_token)
            gnn_encoder_input = self.position_embed(src_input_ast_position) + embed_src_ast_token
            gnn_encoder_input = self.ast_node_emb_dropout(gnn_encoder_input)

            transformer_encoder_output = self.transformer_encoder(transformer_encoder_input, src_mask)
            gnn_encoder_output, node_mask = self.gnn_encoder(gnn_encoder_input, edge_index, node_batch)

            embed_trg_input = self.trg_embed(trg_input)
            decoder_trg_input = self.trg_learnable_embed(embed_trg_input)
            decoder_trg_input = self.text_emb_dropout(decoder_trg_input)
            transformer_decoder_output, _, _ = self.transformer_decoder(transformer_encoder_output, 
                                        gnn_encoder_output, decoder_trg_input, src_mask, node_mask, trg_mask)
            

            transformer_context, gnn_context = self.adaptor(
                transformer_encoder_output, src_mask, gnn_encoder_output, node_mask, transformer_decoder_output)

            final_output = (transformer_context + gnn_context + transformer_decoder_output) / 3

            logits = self.output_layer(final_output) 

            log_probs = F.log_softmax(logits, dim=-1)

            batch_loss = self.loss_function(log_probs, target=trg_truth)
            # NOTE batch loss = sum over all sentence of all tokens in the batch that are not pad!
            return batch_loss

        else:
            return None 
            
       
    def __repr__(self):
        return (f"{self.__class__.__name__}(\n"
                f"\tTransformer_encoder={self.transformer_encoder},\n"
                f"\tGNN_encoder={self.gnn_encoder},\n"
                f"\tTransformer_decoder={self.transformer_decoder},\n"
                f"\tsrc_embed={self.src_embed},\n"
                f"\tcode_learnable_embed={self.code_learnable_embed},\n"  
                f"\tposition_embed={self.position_embed},\n"  
                f"\ttrg_embed={self.trg_embed},\n"
                f"\ttext_learnable_embed={self.trg_learnable_embed},\n"
                f"\tloss_function={self.loss_function})")

    def log_parameters_list(self) -> None:
        """
        Write all model parameters (name, shape) to the log.
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        logger.info("Total parameters number: %d", n_params)
        trainable_parameters = [(name, param) for (name, param) in self.named_parameters() if param.requires_grad]
        for item in trainable_parameters:
            logger.debug("Trainable parameters(name): {0:<60} {1}".format(item[0], str(list(item[1].shape))))
        assert trainable_parameters, "No trainable parameters!"

def build_model(model_cfg: dict=None, vocab_info:dict=None):
    """
    Build the final model according to the configuration.
    """
    logger.info("Build Model...")
    model = Model(model_cfg, vocab_info=vocab_info)
    logger.debug(model)
    model.log_parameters_list()
    logger.info("The model is built.")
    return model

class FaissIndex(object):
    """
    FaissIndex class. factory_template; index_type
    For train index (core: self.index)
    """
    def __init__(self, factory_template:str="IVF256,PQ32", load_index_path:str=None,
                 use_gpu:bool=True, index_type:str="L2") -> None:
        super().__init__()
        self.factory_template = factory_template
        self.gpu_num = faiss.get_num_gpus()
        self.use_gpu = use_gpu and (self.gpu_num > 0)
        logger.warning("use_gpu: {}".format(self.use_gpu))
        self.index_type= index_type
        assert self.index_type in {"L2", "INNER"}
        self._is_trained= False
        if load_index_path != None:
            self.load(index_path=load_index_path)
        
    def load(self, index_path:str) -> faiss.Index:
        self.index = faiss.read_index(index_path)
        if self.use_gpu:
            self.index = faiss.index_cpu_to_all_gpus(self.index)
        self._is_trained = True
    
    @property
    def is_trained(self) -> bool:
        return self._is_trained
    
    def train(self, hidden_representation_path:str) -> None:
        embeddings = np.load(hidden_representation_path, mmap_mode="r")
        total_samples, dimension = embeddings.shape
        logger.info("total samples = {}, dimension = {}".format(total_samples, dimension))
        del embeddings
        # centroids, training_samples = self._get_clustering_parameters(total_samples)
        self.index = self.our_initialize_index(dimension)
        training_embeddinigs = self._get_training_embeddings(hidden_representation_path, total_samples).astype(np.float32)
        self.index.train(training_embeddinigs)
        self._is_trained = True

    def _get_clustering_parameters(self, total_samples: int) -> Tuple[int, int]:
        if 0 < total_samples <= 10 ** 6:
            centroids = int(8 * total_samples ** 0.5)
            training_samples = total_samples
        elif 10 ** 6 < total_samples <= 10 ** 7:
            centroids = 65536
            training_samples = min(total_samples, 64 * centroids)
        else:
            centroids = 262144
            training_samples = min(total_samples, 64 * centroids)
        return centroids, training_samples
    
    def our_initialize_index(self, dimension) -> faiss.Index:
        if self.index_type == "L2":
            index = faiss.index_factory(dimension, "Flat", faiss.METRIC_L2)
        elif self.index_type == "INNER":
            index = faiss.index_factory(dimension, "Flat", faiss.METRIC_INNER_PRODUCT)

        if self.use_gpu:
            index = faiss.index_cpu_to_all_gpus(index)

        return index

    def _initialize_index(self, dimension:int, centroids:int) -> faiss.Index:
        template = re.compile(r"IVF\d*").sub(f"IVF{centroids}", self.factory_template)
        if self.index_type == "L2":
            index = faiss.index_factory(dimension, template, faiss.METRIC_L2)
        elif self.index_type == "INNER":
            index = faiss.index_factory(dimension, template, faiss.METRIC_INNER_PRODUCT)
        else:
            assert False, "Double check index_type!"
        
        if self.use_gpu:
            index = faiss.index_cpu_to_all_gpus(index)
        
        return index
    
    def _get_training_embeddings(self, embeddings_path:str, training_samples: int) -> np.ndarray:
        embeddings = np.load(embeddings_path, mmap_mode="r")
        total_samples = embeddings.shape[0]
        sample_indices = np.random.choice(total_samples, training_samples, replace=False)
        sample_indices.sort()
        training_embeddings = embeddings[sample_indices]
        if self.index_type == "INNER":
            faiss.normalize_L2(training_embeddings)
        return training_embeddings        
    
    def add(self, hidden_representation_path: str, batch_size: int = 10000) -> None:
        assert self.is_trained
        embeddings = np.load(hidden_representation_path)
        total_samples = embeddings.shape[0]
        for i in range(0, total_samples, batch_size):
            start = i 
            end = min(total_samples, i+batch_size)
            batch_embeddings = embeddings[start: end].astype(np.float32)
            if self.index_type == "INNER":
                faiss.normalize_L2(batch_embeddings)
            self.index.add(batch_embeddings)
        del embeddings
    
    def export(self, index_path:str) -> None:
        assert self.is_trained
        if self.use_gpu:
            index = faiss.index_gpu_to_cpu(self.index)
        else:
            index = self.index 
        faiss.write_index(index, index_path)
    
    def search(self, embeddings: np.ndarray, top_k:int=8)-> Tuple[np.ndarray, np.ndarray]:
        assert self.is_trained
        distances, indices = self.index.search(embeddings, k=top_k)
        return distances, indices

    def set_prob(self, nprobe):
        # default nprobe = 1, can try a few more
        # nprobe: search in how many cluster, defualt:1; the bigger nprobe, the result is more accurate, but speed is lower
        self.index.nprobe = nprobe

    @property
    def total(self):
        return self.index.ntotal

class Database(object):
    """
    Initilize with index_path, which is built offline,
    and token path which mapping retrieval indices to token id.
    """
    def __init__(self, index_path:str, token_map_path: str, index_type: str, nprobe:int=16, n_gram:int=1) -> None:
        super().__init__()
        self.index = FaissIndex(load_index_path=index_path, use_gpu=True, index_type=index_type)
        self.index.set_prob(nprobe)
        self.n_gram = n_gram
        self.token_map = self.load_token_mapping(token_map_path)
    
    # staticmethod
    def load_token_mapping(self, token_map_path: str) -> np.ndarray:
        """
        Load token mapping from file.
        """
        if self.n_gram == 1:
            with open(token_map_path) as f:
                token_map = [int(token_id) for token_id in f.readlines()]
            token_map = np.asarray(token_map).astype(np.int32)
        elif self.n_gram == 2:
            with open(token_map_path) as f:
                token_map = []
                for token_id in f.read().splitlines():
                    token_map.append([int(str_id) for str_id in token_id.split(',')])
                token_map = np.asarray(token_map).astype(np.int32)

        return token_map
    
    def search(self, embeddings:np.ndarray, top_k: int=16) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search nearest top_k embeddings from the Faiss index.
        embeddings: np.ndarray (batch_size, d)
        return token_indices: np.ndarray (batch_size, top_k)
        return distances: np.ndarray
        """
        if self.index.index_type == "INNER":
            faiss.normalize_L2(embeddings)
        distances, indices = self.index.search(embeddings, top_k)
        # logger.warning("indices = {}".format(indices))
        token_indices = self.token_map[indices]
        return distances, token_indices

class EnhancedDatabase(Database):
    def __init__(self, index_path:str, token_map_path:str, embedding_path:str, index_type:str, nprobe:int=16, in_memory:bool=True) -> None:
        super().__init__(index_path, token_map_path, index_type, nprobe)
        if in_memory: # load data to memory
            self.embeddings = np.load(embedding_path)
        else:         # the data still on disk
            self.embeddings = np.load(embedding_path, mmap_mode="r")

    def enhanced_search(self, hidden:np.ndarray, top_k:int=8, retrieval_dropout:bool=True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Search nearest top_k embeddings from Faiss index.
        hidden: np.ndarray [batch_size*trg_len, model_dim]
        return distances np.ndarray (batch_size*trg_len, top_k)
        return token_indices: np.ndarray (batch_size*trg_len, top_k)
        return searched_hidden: np.ndarray (batch_size*trg_len, top_k, model_dim)
        """
        if retrieval_dropout:
            distances, indices = self.index.search(hidden, top_k + 1)
            distances = distances[:, 1:]
            indices = indices[:, 1:]
        else:
            distances, indices = self.index.search(hidden, top_k)
        # distances [batch_size*trg_len, top_k]
        # indices [batch_size*trg_len, top_k]

        token_indices = self.token_map[indices]         # token_indices [batch_size*trg_len, top_k]
        searched_hidden = self.embeddings[indices]      # searched_hidden [batch_size*trg_len, top_k, dim]
        return distances, token_indices, searched_hidden

class Kernel(object):
    def __init__(self, index_type:str, n_gram:int) -> None:
        self.index_type = index_type
        self.n_gram = n_gram
        super().__init__()
    
    def similarity(self, distances:torch.Tensor, bandwidth:Union[float, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError
    
    def compute_example_based_distribution(self, distances:torch.Tensor, bandwidth:Union[float, torch.Tensor], 
                                            token_indices:torch.Tensor, vocab_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.n_gram == 1:
            scores = self.similarity(distances, bandwidth)
            # distances[batch_size*trg_len, top_k]
            sparse_distribution = torch.softmax(scores, dim=-1)
            # sparse_distribution [batch_size*trg_len, top_k]        
            zeros = torch.zeros(size=(sparse_distribution.size(0), vocab_size), device=sparse_distribution.device, dtype=sparse_distribution.dtype) 
            distribution = torch.scatter_add(zeros, -1, token_indices, sparse_distribution)
        elif self.n_gram == 2:
            scores = self.similarity(distances, bandwidth)
            sparse_distribution = torch.softmax(scores, dim=-1)
            zeros = torch.zeros(size=(sparse_distribution.size(0), vocab_size), device=sparse_distribution.device, dtype=sparse_distribution.dtype)
            token_indices_token1 = token_indices[:,:,0]
            token_indices_token2 = token_indices[:,:,1]
            # distribution = torch.scatter_add(zeros, -1, token_indices_token1, sparse_distribution)
            distribution = torch.scatter_add(zeros, -1, token_indices_token2, sparse_distribution)
            # distribution = (distribution1 + distribution2) / 2
            # logger.warning(torch.sum(distribution, dim=-1))
        else:
            assert False

        return distribution, sparse_distribution

class GaussianKernel(Kernel):
    def __init__(self, index_type: str, n_gram:int) -> None:
        super().__init__(index_type, n_gram)
    
    def similarity(self, distances: torch.Tensor, bandwidth: Union[float, torch.Tensor]) -> torch.Tensor:
        if self.index_type == "INNER":
            return distances * bandwidth
        elif self.index_type == "L2":
            return - distances / bandwidth
        else:
            assert False

class Retriever(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, hidden: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """
        hidden: [batch_size, trg_len, model_dim]
        logits: [batch_size, trg_len, vocab_size]
        return:
            log_probs: [batch_size, seq_len, vocab_size]
        """
        raise NotImplementedError("The forward method is not implemented in the Retrieval class.")
    
    def detailed_forward(self, hidden:torch.Tensor, logits:torch.Tensor) -> Tuple[torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        hidden: [batch_size, trg_len, model_dim]
        logits: [batch_size, trg_len, vocab_size]
        """
        raise NotImplementedError

class NoRetriever(Retriever):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, hidden: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

class StaticRetriever(Retriever):
    def __init__(self, database:Database, top_k:int, mixing_weight:float, kernel:Kernel, bandwidth:float) -> None:
        super().__init__()
        self.database = database
        self.top_k = top_k 
        self.mixing_weight = mixing_weight
        self.kernel = kernel
        self.bandwidth = bandwidth
    
    def forward(self, hidden:torch.Tensor, logits:torch.Tensor) -> torch.Tensor:
        """
        hidden [batch_size, trg_len, model_dim]
        logits [batch_size, trg_len, trg_vocab_size]
        """
        batch_size, trg_len, model_dim = hidden.size()
        vocab_size = logits.size(-1)
        hidden = hidden.view(batch_size*trg_len, model_dim)
        logits = logits.view(batch_size*trg_len, vocab_size)

        model_based_distribution = F.softmax(logits, dim=-1)
        # model_based_distribution [batch_size*trg_len, trg_vocab_size]

        distances, token_indices = self.database.search(hidden.cpu().numpy(), top_k=self.top_k)
        # logger.warning("token_indices = {}".format(token_indices))
        # logger.warning("distances = {}".format(distances))
        # distances [batch_size*trg_len, top_k] distance
        # token_indices [batch_size*trg_len, top_k] id
        distances = torch.FloatTensor(distances).to(hidden.device)
        token_indices = torch.LongTensor(token_indices).to(hidden.device)
        # distances = distances[:, 2:]
        # token_indices = token_indices[:, 2:]
        # logger.warning("distance = {}".format(distances))
        # logger.warning("distance shape = {}".format(distances.size()))
        # logger.warning("token_indices = {}".format(token_indices))
        # logger.warning("token indices shape = {}".format(token_indices.size()))
        example_based_distribution, _ = self.kernel.compute_example_based_distribution(distances, self.bandwidth, token_indices, vocab_size)
        # example_based_distribution [batch_size*trg_len, trg_vocab_size]

        mixed_distribution = (1 - self.mixing_weight) * model_based_distribution + self.mixing_weight * example_based_distribution

        log_probs = torch.log(mixed_distribution)
        log_probs = log_probs.view(batch_size, trg_len, vocab_size).contiguous()
        # log_probs [batch_size, trg_len, vocab_size]

        analysis = dict()
        analysis["token_indices"] = token_indices
        analysis["model_based_distribution"] = model_based_distribution
        analysis["example_based_distribution"] = example_based_distribution 
        analysis["mixed_distribution"] = mixed_distribution
        analysis["distances"] = distances
        
        return log_probs, analysis, example_based_distribution

def build_retrieval(retrieval_cfg:dict):
    retrieval_type = retrieval_cfg["type"]
    if retrieval_type == "static_retriever":
        database = Database(index_path=retrieval_cfg["index_path"],
                            token_map_path=retrieval_cfg["token_map_path"],
                            index_type=retrieval_cfg["index_type"],
                            n_gram = retrieval_cfg["n_gram"])
        
        assert retrieval_cfg["kernel"] != "Gaussain"
        kernel = GaussianKernel(index_type=retrieval_cfg["index_type"], n_gram=retrieval_cfg["n_gram"])

        retriever = StaticRetriever(database=database,
                    top_k=retrieval_cfg["top_k"],
                    mixing_weight=retrieval_cfg["mixing_weight"],
                    bandwidth=retrieval_cfg["bandwidth"],
                    kernel=kernel)
    else:
        logger.warning("no such retriever {}".format(retrieval_type))

    return retriever

class Adaptor(nn.Module):
    def __init__(self, model_dim) -> None:
        super().__init__()
        self.model_dim = model_dim 

        self.transformer_encoder_adaptor = nn.Linear(model_dim, model_dim)
        self.gnn_encoder_adaptor = nn.Linear(model_dim, model_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, transformer_encoder_output, src_mask, gnn_encoder_output, node_mask, decoder_representation):
        # self.unittest( transformer_encoder_output, src_mask, gnn_encoder_output, node_mask, decoder_representation)
        query = decoder_representation
        # [batch, trg_len, model_dim]
        query = query / math.sqrt(self.model_dim)

        transformer_encoder_adaptor = self.transformer_encoder_adaptor(transformer_encoder_output)
        transformer_scores = torch.matmul(query, transformer_encoder_adaptor.transpose(1,2))
        if src_mask is not None:
            transformer_scores = transformer_scores.masked_fill(~src_mask, float("-inf"))
        transformer_weights = self.softmax(transformer_scores)
        transformer_context = torch.matmul(transformer_weights, transformer_encoder_adaptor)
        # (batch, trg_len, dim)
    
        gnn_encoder_adaptor = self.gnn_encoder_adaptor(gnn_encoder_output)
        gnn_scores = torch.matmul(query, gnn_encoder_adaptor.transpose(1,2))
        if node_mask is not None:
            gnn_scores = gnn_scores.masked_fill(~node_mask.unsqueeze(1), float("-inf"))
        gnn_weights = self.softmax(gnn_scores)
        gnn_context = torch.matmul(gnn_weights, gnn_encoder_adaptor)
        # (batch, trg_len, dim)

        return transformer_context, gnn_context

    def unittest(self, transformer_encoder_output, src_mask, gnn_encoder_output, node_mask, decoder_representation):
        logger.warning("transformer_encoder_output shape = {}".format(transformer_encoder_output.shape))
        logger.warning("src_mask shape = {}".format(src_mask.shape))
        logger.warning("gnn_encoder_output shape = {}".format(gnn_encoder_output.shape))
        logger.warning("node_mask shape = {}".format(node_mask.shape))
        logger.warning("decoder_representation shape = {}".format(decoder_representation.shape))



def build_adaptor(model_dim):
    return Adaptor(model_dim)


if __name__ == "__main__":
    logger = logging.getLogger("")
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.info("Hello! This is Tong Ye's Transformer!")

    import yaml
    from typing import Union 
    def load_config(path: Union[Path,str]="configs/xxx.yaml") -> Dict:
        if isinstance(path, str):
            path = Path(path)
        with path.open("r", encoding="utf-8") as yamlfile:
            cfg = yaml.safe_load(yamlfile)
        return cfg
    
    cfg_file = "test.yaml"
    cfg = load_config(Path(cfg_file))
    vocab_info = {
        "src_vocab": {"size":500, "pad_index":1},
        "position_vocab": {"size":500, "pad_index":1},
        "trg_vocab": {"size":500, "pad_index":1},
    }
    model = build_model(model_cfg=cfg["model"],vocab_info=vocab_info)