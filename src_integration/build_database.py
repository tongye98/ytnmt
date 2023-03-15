"""
build_database module.
FaissIndex, Database, EnhancedDatabase, build_database.
"""
import logging
import torch 
import tqdm 
import faiss 
import re
import numpy as np
from pathlib import Path
from typing import Tuple 
from hashlib import md5
from npy_append_array import NpyAppendArray

import codecs
import pickle 
from test import load_model_checkpoint, resolve_ckpt_path 
from train import load_config, make_logger
from model import build_model
from data import make_data_loader, PAD_ID

logger = logging.getLogger(__name__)

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
    def __init__(self, index_path:str, token_map_path: str, index_type: str, nprobe:int=16) -> None:
        super().__init__()
        self.index = FaissIndex(load_index_path=index_path, use_gpu=True, index_type=index_type)
        self.index.set_prob(nprobe)
        self.token_map = self.load_token_mapping(token_map_path)
    
    @staticmethod
    def load_token_mapping(token_map_path: str) -> np.ndarray:
        """
        Load token mapping from file.
        """
        with open(token_map_path) as f:
            token_map = [int(token_id) for token_id in f.readlines()]
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

def getmd5(sequence: list) -> str:
    sequence = str(sequence)
    return md5(sequence.encode()).hexdigest()

def store_examples(model, hidden_representation_path:str, token_map_path:str,
                   dataset, batch_size:int, seed:int, shuffle:bool,
                   num_workers:int, device:torch.device, use_code_representation:bool):
    """
    Extract hidden states generated by trained model.
    """
    data_loader = make_data_loader(dataset=dataset, sampler_seed=seed, shuffle=False, 
                                batch_size=batch_size, num_workers=num_workers)

    # Create Numpy NPY files by appending on the zero axis.
    npaa = NpyAppendArray(hidden_representation_path)
    token_map_file = open(token_map_path, "w", encoding="utf-8")

    # Statistic Analysis
    total_tokens = 0
    total_sequence = 0
    total_original_tokens = 0

    # don't track gradients during validation
    model.eval()

    for batch_data in tqdm.tqdm(data_loader, desc="Store"):
        batch_data.to(device)

        code_tokens = batch_data.code_tokens
        ast_nodes = batch_data.ast_nodes
        text_tokens = batch_data.text_tokens
        ast_positions = batch_data.ast_positions
        ast_edges = batch_data.ast_edges
        node_batch = batch_data.batch
        # prt = batch_data.ptr
        src_mask = (code_tokens != PAD_ID).unsqueeze(1) # src_mask (batch, 1, code_token_length)
        # src_mask: normal is True; pad is False
        text_tokens_input = text_tokens[:, :-1]
        text_tokens_output = text_tokens[:, 1:]
        # FIXME why is text_tokens output to make the trget mask
        trg_mask = (text_tokens_output != PAD_ID).unsqueeze(1) # trg_mask (batch, 1, trg_length)
        # trg_mask: normal is True, pad is False
        ntokens = (text_tokens_output != PAD_ID).data.sum().item()

        with torch.no_grad():
            penultimate_representation = model(return_type='get_penultimate_representation',  src_input_code_token=code_tokens,
                                src_input_ast_token=ast_nodes, src_input_ast_position=ast_positions,
                                node_batch=node_batch, edge_index=ast_edges,
                                trg_input=text_tokens_input, trg_truth=text_tokens_output,
                                src_mask=src_mask, trg_mask=trg_mask)
            penultimate_representation = penultimate_representation.cpu().numpy().astype("float32")
            # [batch_size, trg_len, model_dim]

        nseqs = code_tokens.shape[0]
        for i in range(nseqs):   
            # for each sentence
            sentence_representation = penultimate_representation[i] #[trg_len, model_dim]
            text_tokens = text_tokens_output[i] #[trg_len]
            for text_token_id, token_representation in zip(text_tokens, sentence_representation):
                total_original_tokens += 1
                if text_token_id != PAD_ID:
                    npaa.append(token_representation[np.newaxis, :])
                    token_map_file.write(f"{text_token_id}\n")
                    total_tokens += 1

        total_sequence += batch_size
            
    del npaa
    token_map_file.close()
    logger.info("Save {} sentences with {} tokens. | Original has {} tokens.".format(total_sequence, total_tokens, total_original_tokens))

def build_database(cfg_file: str):
    """
    The function to store hidden states generated from trained transformer model.
    Handles loading a model from checkpoint, generating hidden states by force decoding and storing them.
    division: which dataset to build database.
    ckpt: use pre-trained model to produce representation.
    hidden_representation_path: where to store token hidden representation.
    token_map_path: where to store corresponding token_map
    index_path: where to store FAISS Index.
    """
    cfg = load_config(Path(cfg_file))
    model_dir = cfg["training"].get("model_dir", None)
    make_logger(Path(model_dir), mode="build_database")

    load_model = cfg["training"].get("load_model", None)
    use_cuda = cfg["training"]["use_cuda"]
    device = torch.device("cuda" if use_cuda else "cpu")
    batch_size = cfg["training"]["batch_size"]
    seed = cfg["training"]["random_seed"]
    shuffle= cfg["training"]["shuffle"]
    num_workers = cfg["training"]["num_workers"]

    # load data
    logger.info("Load data")
    with codecs.open("all_data_stored", 'rb') as f:
        all_data = pickle.load(f)
    train_dataset = all_data["train_datasest"]
    valid_dataset = all_data["valid_dataset"]
    test_dataset = all_data["test_dataset"]
    vocab_info = all_data["vocab_info"]

    # build model
    model = build_model(model_cfg=cfg["model"], vocab_info=vocab_info)

    # when checkpoint is not specified, take latest(best) from model_dir
    ckpt_path = resolve_ckpt_path(None, load_model, Path(model_dir))
    logger.info("ckpt_path = {}".format(ckpt_path))

    # load model checkpoint 
    model_checkpoint = load_model_checkpoint(path=ckpt_path, device=device)

    # restore model and optimizer parameters
    model.load_state_dict(model_checkpoint["model_state"])

    # model to GPU
    if device.type == "cuda":
        model.to(device)

    logger.info("Store train examples...")
    hidden_representation_path = cfg["retrieval"]["hidden_representation_path"]
    token_map_path = cfg["retrieval"]["token_map_path"]
    use_code_representation = cfg["retrieval"]["use_code_representation"]
    store_examples(model, hidden_representation_path=hidden_representation_path, token_map_path=token_map_path,
                    dataset=train_dataset, batch_size=batch_size, seed=seed,
                    shuffle=shuffle, num_workers=num_workers, device=device, use_code_representation=use_code_representation)
    logger.info("Store train examples done!")

    embeddings = np.load(hidden_representation_path, mmap_mode="r")
    total_samples, dimension = embeddings.shape
    logger.info("total samples = {}, dimension = {}".format(total_samples, dimension))

    logger.info("train index...")
    index_path = cfg["retrieval"]["index_path"]
    index_type = cfg["retrieval"]["index_type"]
    index = FaissIndex(index_type=index_type)
    logger.info('start train')
    index.train(hidden_representation_path)
    logger.info('start add')
    index.add(hidden_representation_path)
    logger.info('start export')
    index.export(index_path)
    logger.info("train index done!")

    del index

if __name__ == "__main__":
    cfg = "test.yaml"
    build_database(cfg)