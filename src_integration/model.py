import math 
import logging 
import numpy as np
from pathlib import Path
from typing import Dict, Tuple 
import torch 
import torch.nn as nn 
from torch import Tensor 
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import SAGEConv, GCNConv, GATConv

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
        self.dropout= nn.Dropout(0.2)
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
        node_encode = self.dropout(node_enc_) + residual
        return node_encode

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
    
    def context_representation(self, penultimate:Tensor, encoder_output:Tensor, src_mask:Tensor, trg_mask:Tensor) -> Tensor:
        """
        Get the hidden state for search purpose.
        The hidden state means the token semantic.
        """
        residual = penultimate
        if self.layer_norm_position == 'pre':
            penultimate = self.layer_norm(penultimate)
        self_attention_output, _ = self.trg_trg_attention(penultimate, penultimate, penultimate, trg_mask)
        cross_attention_input = self.dropout(self_attention_output) + residual

        if self.layer_norm_position == 'post':
            cross_attention_input = self.layer_norm(cross_attention_input)
        
        cross_residual = cross_attention_input
        if self.layer_norm_position == 'pre':
            cross_attention_input = self.layer_norm2(cross_attention_input)
        cross_attention_output, cross_attention_weight = self.src_trg_attention(encoder_output, encoder_output, cross_attention_input, src_mask)
        feedforward_input = self.dropout(cross_attention_output) + cross_residual

        if self.layer_norm_position == "post":
            feedforward_input = self.layer_norm2(feedforward_input)
            
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
        
        penultimate_representation = self.layers[-1].context_representation(penultimate, transformer_encoder_output, src_mask, trg_mask)

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
                trg_mask:Tensor=None,):
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

        elif return_type == "encode_decode":
            pass 
    
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