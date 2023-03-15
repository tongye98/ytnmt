import json
import codecs 
import functools
from tqdm import tqdm  
import numpy as np 
from pathlib import Path 
from typing import Tuple, Union, List, Dict
from collections import Counter
import operator
import unicodedata
import logging
import pickle 
import torch 
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

"""
Global constants
"""
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"

UNK_ID = 0
PAD_ID = 1
BOS_ID = 2
EOS_ID = 3

logger = logging.getLogger(__name__)

class Vocabulary(object):
    """
    Vocabulary class mapping between tokens and indices.
    """
    def __init__(self, tokens: List[str]) -> None:
        "Create  vocabulary from list of tokens. :param tokens: list of tokens"

        self.specials = [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]
        self._stoi: Dict[str, int] = {} # string to index
        self._itos: List[str] = []      # index to string

        # construct vocabulary
        self.add_tokens(tokens=self.specials + tokens)
        assert len(self._stoi) == len(self._itos)

        # assign special after stoi and itos are built
        self.pad_index = self.lookup(PAD_TOKEN)
        self.unk_index = self.lookup(UNK_TOKEN)
        self.bos_index = self.lookup(BOS_TOKEN)
        self.eos_index = self.lookup(EOS_TOKEN)
        assert self.pad_index == PAD_ID
        assert self.unk_index == UNK_ID
        assert self.bos_index == BOS_ID
        assert self.eos_index == EOS_ID
        assert self._itos[UNK_ID] == UNK_TOKEN
    
    def lookup(self, token: str) -> int:
        "look up the encoding dictionary"
        return self._stoi.get(token, UNK_ID) 
    
    def add_tokens(self, tokens:List[str]) -> None:
        for token in tokens:
            # token = self.normalize(token)
            new_index = len(self._itos)
            # add to vocabulary if not already there
            if token not in self._itos:
                self._itos.append(token)
                self._stoi[token] = new_index
    
    def is_unk(self,token:str) -> bool:
        """
        Check whether a token is covered by the vocabulary.
        """
        return self.lookup(token) == UNK_ID
    
    def to_file(self, file_path: Path) -> None:
        def write_list_to_file(file_path:Path, array:List[str]) -> None:
            """
            write list of str to file.
            """
            with file_path.open("w", encoding="utf-8") as fg:
                for item in array:
                    fg.write(f"{item}\n")
        
        write_list_to_file(file_path, self._itos)
    
    def __len__(self) -> int:
        return len(self._itos)
    
    @staticmethod
    def normalize(token) -> str:
        return unicodedata.normalize('NFD', token)
    
    def array_to_sentence(self, array: np.ndarray, cut_at_eos: bool=True, skip_pad: bool=True) -> List[str]:
        """
        Convert an array of IDs to a sentences (list of tokens).
        array: 1D array containing indices
        Note: when cut_at_eos=True, sentence final token is </s>.
        """
        sentence = []
        for i in array:
            token = self._itos[i]
            if skip_pad and token == PAD_TOKEN:
                continue
            sentence.append(token)
            if cut_at_eos and token == EOS_TOKEN:
                break
        
        return [token for token in sentence if token not in self.specials]

    def arrays_to_sentences(self, arrays: np.ndarray, cut_at_eos: bool=True, skip_pad: bool=True) -> List[List[str]]:
        """
        Convert multiple arrays containing sequences of token IDs to their sentences.
        arrays: 2D array containing indices.
        return: list of list of tokens.
        """
        return [self.array_to_sentence(array=array, cut_at_eos=cut_at_eos, skip_pad=skip_pad) for array in arrays]

    def tokens_to_ids(self, tokens:List[str], bos:bool=False, eos:bool=False):
        """
        Return sentences_ids List[id].
        """
        tokens_ids = [self.lookup(token) for token in tokens]
        if bos is True:
            tokens_ids = [self.bos_index] + tokens_ids
        if eos is True:
            tokens_ids = tokens_ids + [self.eos_index]

        return tokens_ids

    def log_vocab(self, number:int) -> str:
        "First how many number of tokens in Vocabulary."
        return " ".join(f"({id}) {token}" for id, token in enumerate(self._itos[:number]))

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(len={self.__len__()}, "
                f"specials={self.specials})")

def read_data_from_file(path:Path):
    with codecs.open(path, encoding='utf-8') as f:
        raw_data = json.load(f)
    
    data = []
    dataset_truth = []
    for item in tqdm(raw_data, desc="Extract data from file..."):
        code_tokens = item['code'].split()
        text_truth = item["text"]
        dataset_truth.append(text_truth)
        text_tokens = item['text'].split()
        ast_nodes = list(eval(item['ast']['nodes']))
        ast_positions = list(eval(item['ast']['poses']))
        assert len(ast_nodes) == len(ast_positions), "ast nodes != ast positions"

        ast_edges = np.array(eval(item['ast']['edges']))
        reversed_edges = np.array([ast_edges[1,:], ast_edges[0,:]])
        ast_edges = np.concatenate([ast_edges, reversed_edges], axis=-1)

        data_item = {
            "code_tokens": code_tokens,
            "text_tokens": text_tokens,
            "ast_nodes": ast_nodes,
            "ast_edges": ast_edges,
            "ast_positions": ast_positions,
        }
        data.append(data_item)

    return data, dataset_truth

def log_vocab_info(code_vocab,text_vocab, position_vocab):
    """logging vocabulary information"""
    logger.info("code vocab length = {}".format(len(code_vocab)))
    logger.info("text vocab length = {}".format(len(text_vocab)))
    logger.info("position vocab length = {}".format(len(position_vocab)))
    return None 

def log_data_info(code_tokens, ast_tokens, text_tokens, position_tokens, ast_edges):
    """logging data statistics"""
    assert len(code_tokens) == len(ast_tokens) == len(text_tokens) == len(position_tokens) == len(ast_edges), "data need double check!"
    len_code_token = len_ast_token = len_text_token = len_position_token = len_ast_edge= 0
    for code_token, ast_token, text_token, position_token, ast_edge in zip(code_tokens, ast_tokens, text_tokens, position_tokens, ast_edges):
        len_code_token += len(code_token)
        len_ast_token += len(ast_token)
        len_text_token += len(text_token)
        len_position_token += len(position_token)
        len_ast_edge += ast_edge.shape[1]
    
    logger.info("average code tokens = {}".format(len_code_token / len(code_tokens)))
    logger.info("average ast tokens = {}".format(len_ast_token / len(ast_tokens)))
    logger.info("average text tokens = {}".format(len_text_token / len(text_tokens)))
    logger.info("average position tokens = {}".format(len_position_token / len(position_tokens)))
    logger.info("average ast edges = {}".format(len_ast_edge / len(ast_edges)))
    return None 

def build_vocabulary(data_cfg:dict, datasets):
    """
    code_vocabulary: {code_token, ast_token}
    positon_vocabulary: {position}
    text_vocabulary: {text_token}
    """
    def flatten(array):
        # flatten a nested 2D list.
        return functools.reduce(operator.iconcat, array, [])

    def sort_and_cut(counter, max_size:int, min_freq:int) -> List[str]:
        """
        Cut counter to most frequent, sorted numerically and alphabetically.
        return: list of valid tokens
        """
        if min_freq > -1:
            counter = Counter({t: c for t, c in counter.items() if c >= min_freq})
        
        # sort by frequency, then alphabetically
        tokens_and_frequencies = sorted(counter.items(),key=lambda tup: tup[0])
        tokens_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        # cut off
        vocab_tokens = [i[0] for i in tokens_and_frequencies[:max_size]]
        assert len(vocab_tokens) <= max_size, "vocab tokens must <= max_size."
        return vocab_tokens
    
    code_min_freq = data_cfg['code'].get("vocab_min_freq", 1)
    code_max_size = data_cfg['code'].get("vocab_max_size", -1)
    assert code_max_size > 0 and code_min_freq > 0
    text_min_freq = data_cfg['text'].get("vocab_min_freq", 1)
    text_max_size = data_cfg['text'].get("vocab_max_size", -1)
    assert text_max_size > 0 and text_min_freq > 0
    position_min_freq = data_cfg["position"].get("vocab_min_freq", 1)
    position_max_size = data_cfg["position"].get("vocab_max_size", -1)
    assert position_max_size > 0 and position_min_freq > 0

    code_tokens = []  # list of list [[], [], []]  already double checked!
    ast_tokens = []   # list of list [[], [], []]
    text_tokens = []  # list of list [[], [], []]
    position_tokens = [] # list of list [[], [], []]
    ast_edges = [] # list of  ndarray [ndarray, ndarray, ndarray]
    for idx, dataset in enumerate(datasets):
        for item in tqdm(dataset, desc='build vocabulary for #{} dataset...'.format(idx)):
            code_tokens.append(item['code_tokens'])
            ast_tokens.append(item['ast_nodes'])
            text_tokens.append(item['text_tokens'])
            position_tokens.append(item['ast_positions'])
            ast_edges.append(item['ast_edges'])

    log_data_info(code_tokens, ast_tokens, text_tokens, position_tokens, ast_edges)

    # code token and ast tokens share a same vocabulary
    code_tokens.extend(ast_tokens)
    code_counter = Counter(flatten(code_tokens))
    text_counter = Counter(flatten(text_tokens))
    position_counter = Counter(flatten(position_tokens))

    code_unique_tokens = sort_and_cut(code_counter, code_max_size, code_min_freq)
    text_unique_tokens = sort_and_cut(text_counter, text_max_size, text_min_freq)
    position_unique_tokens = sort_and_cut(position_counter, position_max_size, position_min_freq)

    code_vocab = Vocabulary(code_unique_tokens)
    text_vocab = Vocabulary(text_unique_tokens)
    position_vocab = Vocabulary(position_unique_tokens)

    log_vocab_info(code_vocab, text_vocab, position_vocab)

    return code_vocab, text_vocab, position_vocab

def load_data(data_cfg: dict):
    """
    Load train, valid and test data as specified in configuration.
    """
    train_data_path = Path(data_cfg.get("train_data_path", None))
    valid_data_path = Path(data_cfg.get("valid_data_path", None))
    test_data_path = Path(data_cfg.get("test_data_path", None))
    assert train_data_path.is_file() and valid_data_path.is_file() and test_data_path.is_file(), \
    "train or valid or test path is not a file."

    train_data, train_data_truth = read_data_from_file(train_data_path)
    valid_data, valid_data_truth = read_data_from_file(valid_data_path)
    test_data, test_data_truth = read_data_from_file(test_data_path)

    code_vocab, text_vocab, position_vocab = build_vocabulary(data_cfg, [train_data, valid_data])

    vocab_info = {
        "src_vocab": {"self":code_vocab ,"size": len(code_vocab), "pad_index": code_vocab.pad_index},
        "position_vocab": {"self":position_vocab, "size":len(position_vocab), "pad_index": position_vocab.pad_index},
        "trg_vocab": {"self":text_vocab, "size":len(text_vocab), "pad_index": position_vocab.pad_index}
    }

    train_data_id = token2id(train_data, code_vocab, text_vocab, position_vocab, data_cfg)
    valid_data_id = token2id(valid_data, code_vocab, text_vocab, position_vocab, data_cfg)
    test_data_id = token2id(test_data, code_vocab, text_vocab, position_vocab, data_cfg)

    train_dataset = OurDataset(train_data_id, train_data_truth)
    valid_dataset = OurDataset(valid_data_id, valid_data_truth)
    test_dataset = OurDataset(test_data_id, test_data_truth)

    all_data = {"train_datasest": train_dataset, "valid_dataset":valid_dataset,
                "test_dataset": test_dataset, "vocab_info": vocab_info}
    
    with open("all_data_stored", 'wb') as f:
        pickle.dump(all_data, f)

    return train_dataset, valid_dataset, test_dataset, vocab_info

def token2id(data, code_vocab:Vocabulary, text_vocab:Vocabulary, position_vocab:Vocabulary, data_cfg):
    """
    token to id. And truc and pad for code_tokens_id and text_tokens_id.
    """
    code_token_max_len = data_cfg["code"].get("code_token_max_len", 200)
    text_token_max_len = data_cfg["text"].get("text_token_max_len", 40)
    # data [dict, dict, ...]
    data_id = []
    for item in tqdm(data, desc="token2id"):
        data_item_id = {}
        data_item_id["code_tokens_id"] = truc_pad(code_vocab.tokens_to_ids(item["code_tokens"], bos=False, eos=False), code_token_max_len)
        data_item_id["ast_nodes_id"] = code_vocab.tokens_to_ids(item["ast_nodes"], bos=False, eos=False)
        data_item_id["text_tokens_id"] = truc_pad(text_vocab.tokens_to_ids(item["text_tokens"], bos=True, eos=True), text_token_max_len)
        data_item_id["ast_positions_id"] = position_vocab.tokens_to_ids(item["ast_positions"], bos=False, eos=False)
        data_item_id["ast_edgs_id"]= item["ast_edges"]
        data_id.append(data_item_id)

    return data_id

def truc_pad(data, token_max_len):
    truc_data = data[:token_max_len]
    pad_number = token_max_len - len(truc_data)
    assert pad_number >=0, "pad number must >=0!"
    pad_data = truc_data + [PAD_ID] * pad_number
    return pad_data

class OurDataset(Dataset):
    def __init__(self, data_id, data_truth) -> None:
        super().__init__()
        self.data_id = data_id  # data_id [dict, dict, ...]
        self.target_truth = data_truth
    
    def __getitem__(self, index):
        data_item = self.data_id[index]
        code_tokens_id = data_item["code_tokens_id"]
        ast_nodes_id = data_item["ast_nodes_id"]
        text_tokens_id = data_item["text_tokens_id"]
        ast_positions_id = data_item["ast_positions_id"]
        ast_edges_id = data_item["ast_edgs_id"]

        return OurData(code_tokens=torch.tensor(code_tokens_id, dtype=torch.long), 
                       ast_nodes=torch.tensor(ast_nodes_id, dtype=torch.long),
                       text_tokens=torch.tensor(text_tokens_id, dtype=torch.long), 
                       ast_positions=torch.tensor(ast_positions_id, dtype=torch.long),
                       ast_edges=torch.tensor(ast_edges_id, dtype=torch.long))
    
    def __len__(self) -> int:
        return len(self.data_id)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(len={self.__len__()}"

class OurData(Data):
    def __init__(self, code_tokens, ast_nodes, text_tokens, ast_positions, ast_edges, *args, **kwargs):
        super().__init__()
        
        self.code_tokens = code_tokens
        self.ast_nodes = ast_nodes
        self.text_tokens = text_tokens 
        self.ast_positions = ast_positions
        self.ast_edges = ast_edges

    def __inc__(self, key, value, *args, **kwargs):
        if key == "ast_edges":
            return self.ast_nodes.size(0)
        else: 
            return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "ast_edges":
            return -1
        elif key == "code_tokens" or key == "text_tokens":
            return None 
        elif key == "ast_nodes" or key == "ast_positions":
            return 0
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)

def make_data_loader(dataset: OurDataset, sampler_seed, shuffle, batch_size, num_workers, mode=None) -> DataLoader:
    """
    Return a torch DataLoader.
    """
    assert isinstance(dataset, Dataset), "For pytorch, dataset is based on torch.utils.data.Dataset"

    if mode == "train" and shuffle is True:
        generator = torch.Generator()
        generator.manual_seed(sampler_seed)
        sampler = RandomSampler(dataset, replacement=False, generator=generator)
    else:
        sampler = SequentialSampler(dataset)
    
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, 
                      num_workers=num_workers, pin_memory=True)

if __name__ == "__main__": 
    logger = logging.getLogger("")
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.info("Hello! This is Tong Ye's Transformer!")

    import yaml
    def load_config(path: Union[Path,str]="configs/xxx.yaml"):
        if isinstance(path, str):
            path = Path(path)
        with path.open("r", encoding="utf-8") as yamlfile:
            cfg = yaml.safe_load(yamlfile)
        return cfg
    
    cfg_file = "test.yaml"
    cfg = load_config(Path(cfg_file))
    train_dataset, valid_dataset, test_dataset = load_data(data_cfg=cfg["data"])

    assert False
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)
    batch_data = next(iter(train_loader))

    # check batch_data
    logger.info(batch_data)
    logger.info("code_tokens = {}".format(batch_data.code_tokens))
    logger.info("ast_nodes = {}".format(batch_data.ast_nodes))
    logger.info("text_tokens = {}".format(batch_data.text_tokens))
    logger.info("ast_positions = {}".format(batch_data.ast_positions))
    logger.info("ast_edges = {}".format(batch_data.ast_edges))
    logger.info("batch = {}".format(batch_data.batch))
    logger.info("ptr = {}".format(batch_data.ptr))