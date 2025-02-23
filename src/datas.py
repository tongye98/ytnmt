"""
Data module.
Implementation: load_data(); make_data_iter(); collate_fn(); Batch
"""
import logging
import torch 
import numpy as np
from src.tokenizers import build_tokenizer
from src.datasets import build_dataset
from src.vocabulary import build_vocab
from src.helps import ConfigurationError, log_data_info
from src.constants import PAD_ID
from torch.utils.data import Dataset, Sampler, DataLoader
from torch.utils.data import SequentialSampler, RandomSampler, BatchSampler
from typing import List, Union, Tuple, Iterator, Iterable

logger = logging.getLogger(__name__)

def load_data(data_cfg: dict):
    """
    Load train, dev and test data as specified in configuration.
    Vocabularies are created from the training dataset with a limit of 'vocab_limit' tokens
    and a minimum token frequency of "vocab_min_freq".
    The training data is cut off to include sentences up to "max_sentence_length"
    on source and target side.
    return:
        - src_vocab: source vocabulary
        - trg_vocab: traget vocabulary
        - train_data: training dataset
        - dev_data: development dataset
        - test_data: test dataset
    """
    src_cfg = data_cfg["src"]
    trg_cfg = data_cfg["trg"]

    src_language = src_cfg["language"]
    trg_language = trg_cfg["language"]

    train_data_path = data_cfg.get("train_data_path", None)
    dev_data_path = data_cfg.get("dev_data_path", None)
    test_data_path = data_cfg.get("test_data_path", None)
    assert train_data_path is not None
    assert dev_data_path is not None 
    assert test_data_path is not None 
    
    # build tokenizer
    logger.info("Build tokenizer...")
    tokenizer = build_tokenizer(data_cfg)

    dataset_type = data_cfg.get("dataset_type", "plain")

    # train data 
    train_data = None 
    logger.info("Loading train dataset...")
    train_data = build_dataset(dataset_type=dataset_type,path=train_data_path, split_mode="train",
                                src_language=src_language, trg_language=trg_language, tokenizer=tokenizer)
    # dev data
    dev_data = None
    logger.info("Loading dev dataset...")
    dev_data = build_dataset(dataset_type=dataset_type, path=dev_data_path, split_mode="dev",
                             src_language=src_language, trg_language=trg_language, tokenizer=tokenizer)
    # test data
    logger.info("Loading test dataset...")
    test_data = build_dataset(dataset_type=dataset_type, path=test_data_path, split_mode="test",
                              src_language=src_language, trg_language=trg_language, tokenizer=tokenizer)
    
    # build vocabulary
    logger.info("Building vocabulary...")
    src_vocab, trg_vocab = build_vocab(data_cfg, datasets=[train_data, dev_data])

    train_data.tokernized_data_to_ids(src_vocab, trg_vocab)
    dev_data.tokernized_data_to_ids(src_vocab, trg_vocab)
    test_data.tokernized_data_to_ids(src_vocab, trg_vocab)

    logger.info("Dataset has loaded.")
    # log dataset and vocabulary information.
    log_data_info(train_data, dev_data, test_data, src_vocab, trg_vocab)

    return train_data, dev_data, test_data, src_vocab, trg_vocab

def make_data_iter(dataset:Dataset, sampler_seed, shuffle, batch_type,
                   batch_size, num_workers) -> DataLoader:
    """
    Return a torch DataLoader.
    """
    assert isinstance(dataset, Dataset), "For pytorch, dataset is based on torch.utils.data.Dataset"

    if dataset.split_mode == "train" and shuffle is True:
        generator = torch.Generator()
        generator.manual_seed(sampler_seed)
        sampler = RandomSampler(dataset, replacement=False, generator=generator)
        # sampler = SortedBatchSampler(dataset.lengths(), batch_size, shuffle=True)
    else:
        sampler = SequentialSampler(dataset)
    
    if batch_type == "sentence":
        batch_sampler = SentenceBatchSampler(sampler, batch_size=batch_size, drop_last=False)
    elif batch_type == "token":
        raise NotImplementedError("This batch_type Not Implementation.")
    else:
        raise ConfigurationError("Invalid batch_type.")

    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
                    pin_memory=True, collate_fn=collate_fn)
    
class SortedBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        lengths = np.array(
            [(-l[0], -l[1], np.random.random()) for l in self.lengths],
            dtype=[('l1', np.int_), ('l2', np.int_), ('rand', np.float_)]
        )
        indices = np.argsort(lengths, order=('l1', 'l2', 'rand'))
        batches = [indices[i:i + self.batch_size]
                   for i in range(0, len(indices), self.batch_size)]
        if self.shuffle:
            np.random.shuffle(batches)
        return iter([i for batch in batches for i in batch])

    def __len__(self):
        return len(self.lengths)

class SentenceBatchSampler(BatchSampler):
    """
    Classical BatchSampler.
    """
    def __init__(self, sampler: Union[Sampler[int], Iterable[int]], batch_size: int, drop_last: bool) -> None:
        super().__init__(sampler, batch_size, drop_last)
    
    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch
    
    def __len__(self) -> int:
        return super().__len__()

def collate_fn(batch: List[Tuple]):
    """
    Custom collate function.
    Note: you can stack batch and any operation on batch.
    DataLoader every iter result is collate_fn's return value -> Batch.
    :param batch [(src,trg), (src,trg), ...]
    """
    src_list, trg_list= zip(*batch)  # src_list: Tuple[List[id]] || trg_list: Tuple[List[id]]
    assert len(src_list) == len(trg_list)
    return Batch(src_list, trg_list)

class Batch(object):
    """
    Object for holding a batch of data with mask during training.
    Input is yield from 'collate_fn()' called by torch.data.utils.DataLoader.
    """
    def __init__(self, src, trg):
        src_lengths = [len(sentence) for sentence in src] # not include <bos>, include <eos>
        max_src_len = max(src_lengths)
        trg_lengths = [len(sentence) for sentence in trg] # include <bos> and <eos>
        max_trg_len = max(trg_lengths)

        padded_src_sentences = []
        for sentence in src:
            pad_number = max_src_len - len(sentence)
            assert pad_number >= 0, "pad number must >= 0!"
            padded_src_sentences.append(sentence + [PAD_ID] * pad_number)
        
        padded_trg_sentences = []
        for sentence in trg:
            pad_number = max_trg_len -len(sentence)
            assert pad_number >= 0, "pad number must >=0!"
            padded_trg_sentences.append(sentence + [PAD_ID] * pad_number)
        
        self.src = torch.tensor(padded_src_sentences).long()
        self.src_lengths = torch.tensor(src_lengths).long() # src_lengths Shape: (batch)
        self.src_mask = (self.src != PAD_ID).unsqueeze(1)
        # src_mask unpad is true, pad is false; Shape:(batch, 1, pad_src_length)
        self.nseqs = self.src.size(0)
        
        self.trg = torch.tensor(padded_trg_sentences).long()
        self.trg_input = self.trg[:, :-1]
        self.trg_truth = self.trg[:, 1:]  # self.trg_truth is used for loss computation
        self.trg_lengths = torch.tensor(trg_lengths).long() - 1 # original trg length + 1
        self.trg_mask = (self.trg_truth != PAD_ID).unsqueeze(1) # Shape: [batch_size, 1, trg_length]
        self.ntokens = (self.trg_truth != PAD_ID).data.sum().item()

    def move2cuda(self, device:torch.device):
        """Move batch data to GPU"""
        assert isinstance(device, torch.device)
        assert device.type == "cuda", "In move2data: device type != cuda."

        self.src = self.src.to(device, non_blocking=True)
        self.src_lengths = self.src_lengths.to(device, non_blocking=True)
        self.src_mask = self.src_mask.to(device, non_blocking=True)

        self.trg_input = self.trg_input.to(device, non_blocking=True)
        self.trg_truth = self.trg_truth.to(device, non_blocking=True)
        self.trg_lengths = self.trg_lengths.to(device, non_blocking=True)
        self.trg_mask = self.trg_mask.to(device, non_blocking=True)

    def normalize(self, tensor, normalization):
        """
        Normalizes batch tensor (i.e. loss)
        """
        if normalization == "sum":
            return tensor 
        elif normalization == "batch":
            normalizer = self.nseqs
        elif normalization == "tokens":
            normalizer = self.ntokens
        elif normalization == "none":
            normalizer = 1
        
        norm_tensor = tensor / normalizer
        return norm_tensor
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(nseqs={self.nseqs}, ntokens={self.ntokens}.)")
    