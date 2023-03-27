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
from train import load_config, make_logger
from model import build_model, FaissIndex
from data import make_data_loader, PAD_ID

logger = logging.getLogger(__name__)

def resolve_ckpt_path(ckpt_path:str, load_model:str, model_dir:Path) -> Path:
    """
    Resolve checkpoint path
    First choose ckpt_path, then choos load_model, 
    then choose model_dir/best.ckpt, final choose model_dir/latest.ckpt
    """
    if ckpt_path is None:
        logger.warning("ckpt_path is not specified.")
        if load_model is None:
            if (model_dir / "best.ckpt").is_file():
                ckpt_path = model_dir / "best.ckpt"
                logger.warning("use best ckpt in model dir!")
            else:
                logger.warning("No ckpt_path, no load_model, no best_model, Please Check!")
                ckpt_path = model_dir / "latest.ckpt"
                logger.warning("use latest ckpt in model dir!")
        else:
            logger.warning("use load_model item in config yaml.")
            ckpt_path = Path(load_model)
    return Path(ckpt_path)

def load_model_checkpoint(path:Path, device:torch.device):
    """
    Load model from saved model checkpoint
    """
    assert path.is_file(), f"model checkpoint {path} not found!"
    model_checkpoint = torch.load(path.as_posix(), map_location=device)
    logger.info("Load model from %s.", path.resolve())
    return model_checkpoint

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
        n_gram = 2
        for i in range(nseqs):   
            # for each sentence
            if n_gram == 1:
                sentence_representation = penultimate_representation[i] #[trg_len, model_dim]
                text_tokens = text_tokens_output[i] #[trg_len]
                for text_token_id, token_representation in zip(text_tokens, sentence_representation):
                    total_original_tokens += 1
                    if text_token_id != PAD_ID:
                        npaa.append(token_representation[np.newaxis, :])
                        token_map_file.write(f"{text_token_id}\n")
                        total_tokens += 1
            elif n_gram == 2:
                # logger.warning("n_gram = 2")
                sentence_representation = penultimate_representation[i] # [trg_len, model_dim]
                text_tokens = text_tokens_output[i] # [trg_len]
                trg_len = sentence_representation.shape[0]

                for i in range(1, trg_len):
                    total_original_tokens += 1
                    current_token_representation = sentence_representation[i] # [model_dim]
                    prevent_token_representation = sentence_representation[i-1] # [model_dim]
                    mean_token_representation = (current_token_representation + prevent_token_representation) / 2
                    
                    current_text_token_id = text_tokens[i]
                    prevent_text_token_id = text_tokens[i-1]
                    if current_text_token_id != PAD_ID or prevent_text_token_id != PAD_ID:
                        npaa.append(mean_token_representation[np.newaxis, :])
                        token_map_file.write("{},{}\n".format(prevent_text_token_id, current_text_token_id))
                        total_tokens += 1
            else:
                assert False

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
    with codecs.open("../models/codescribe_java/no_gnn_residual/all_data_stored", 'rb') as f:
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
    cfg = "../models/codescribe_java/no_gnn_residual/no_residual.yaml"
    build_database(cfg)