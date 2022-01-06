from transformers import GPT2TokenizerFast, BertTokenizerFast
from torch.utils.data import TensorDataset, DataLoader
from typing import Sequence
import tensorflow as tf
import pandas as pd
import torch

class Preprocesser:
    def __init__(self):
        self.RANDOM_SEED = 10
        # HyperParam
        self.batch_size = 16
        self.max_len = 201  # train_x : 184 | train_Y : 201 | val_x : 127 | val_Y : 148
        # data
        self.data_num = 3798  # train - 3362, val - 436
        self.PREMODEL_NAME = "byeongal/Ko-DialoGPT"
        self.COMPRESS_MODEL_NAME = "monologg/kobert"
        self.tokenizer = GPT2TokenizerFast.from_pretrained("tokenizer")
        self.compress_tokenizer = BertTokenizerFast.from_pretrained(self.COMPRESS_MODEL_NAME, use_cache=True,
                                                                    cache_dir="./tokenizer/compress_tokenizer")
        self.vocab_size = self.tokenizer.vocab_size

    def getTFData(self, is_train: bool) -> tf.data.Dataset:
        # data => dialogue : S1</s>S2</s> | response : R1</s>
        if is_train:
            data = pd.read_csv("data/train.txt", sep="\t", names=["dialogue", "response"], header=0)
        else:
            data = pd.read_csv("data/val.txt", sep="\t", names=["dialogue", "response"], header=0)

        x = self.tokenizer.batch_encode_plus(data["dialogue"].to_list(), return_tensors="tf",
                                             max_length=self.max_len, padding="max_length", truncation=True)
        encoded_x = dict()
        for key, value in x.items():
            encoded_x[key] = value

        Y = self.tokenizer.batch_encode_plus((data["dialogue"] + data["response"]).to_list(), return_tensors="tf",
                                             max_length=self.max_len, padding="max_length", truncation=True)["input_ids"]
        return tf.data.Dataset.from_tensor_slices((encoded_x, Y)).batch(self.batch_size).shuffle(256, seed=self.RANDOM_SEED)

    def getTorchData(self, is_train: bool, device: str) -> DataLoader:
        torch.manual_seed(self.RANDOM_SEED)
        torch.cuda.manual_seed(self.RANDOM_SEED)
        if is_train:
            data = pd.read_csv("data/train.txt", sep="\t", names=["dialogue", "response"], header=0)
        else:
            data = pd.read_csv("data/val.txt", sep="\t", names=["dialogue", "response"], header=0)
        x = self.tokenizer.batch_encode_plus(data["dialogue"].to_list(), return_tensors="pt",
                                             max_length=self.max_len, padding="max_length", truncation=True)
        Y = self.tokenizer.batch_encode_plus((data["dialogue"] + data["response"]).to_list(),
                                             return_tensors="pt", max_length=self.max_len, padding="max_length", truncation=True)["input_ids"]
        dataset = TensorDataset(x["input_ids"].to(device), x["attention_mask"].to(device), Y.to(device))

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def encoding(self, text: str, return_tensors: str = "pt") -> tf.Tensor:
        return self.tokenizer.encode(text, return_tensors=return_tensors)

    def decoding(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)
