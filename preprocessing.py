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
                                                                    cache_dir="tokenizer/compress_tokenizer")
        self.vocab_size = self.tokenizer.vocab_size

    def getTrainData(self) -> tf.data.Dataset:
        # data's dialogue : S1</s>S2</s> | response : R1</s>
        train_data = pd.read_csv("data/train.txt", sep="\t", names=["dialogue", "response"], header=0)

        train_x = self.tokenizer.batch_encode_plus(train_data["dialogue"].to_list(), return_tensors="tf",
                                                   max_length=self.max_len, padding="max_length", truncation=True)
        encoded_train_x = dict()
        for key, value in train_x.items():
            encoded_train_x[key] = value

        train_Y = self.tokenizer.batch_encode_plus((train_data["dialogue"] + train_data["response"]).to_list(), return_tensors="tf",
                                                   max_length=self.max_len, padding="max_length", truncation=True)["input_ids"]
        return tf.data.Dataset.from_tensor_slices((encoded_train_x, train_Y)).batch(self.batch_size).shuffle(256, seed=self.RANDOM_SEED)

    def getValidationData(self) -> tf.data.Dataset:
        val_data = pd.read_csv("data/val.txt", sep="\t", names=["dialogue", "response"], header=0)

        val_x = self.tokenizer.batch_encode_plus(val_data["dialogue"].to_list(), return_tensors="tf",
                                                 max_length=self.max_len, padding="max_length", truncation=True)
        encoded_val_x = dict()
        for key, value in val_x.items():
            encoded_val_x[key] = value

        val_Y = self.tokenizer.batch_encode_plus((val_data["dialogue"] + val_data["response"]).to_list(), return_tensors="tf",
                                                 max_length=self.max_len, padding="max_length", truncation=True)["input_ids"]
        return tf.data.Dataset.from_tensor_slices((encoded_val_x, val_Y)).batch(self.batch_size).shuffle(256, seed=self.RANDOM_SEED)

    def getTorchTrainData(self) -> DataLoader:
        torch.manual_seed(self.RANDOM_SEED)
        torch.cuda.manual_seed(self.RANDOM_SEED)

        train_data = pd.read_csv("data/train.txt", sep="\t", names=["dialogue", "response"], header=0)

        train_x = self.tokenizer.batch_encode_plus(train_data["dialogue"].to_list(), return_tensors="pt",
                                                   max_length=self.max_len, padding="max_length", truncation=True)
        train_Y = self.tokenizer.batch_encode_plus((train_data["dialogue"] + train_data["response"]).to_list(),
                                                   return_tensors="pt", max_length=self.max_len, padding="max_length", truncation=True)["input_ids"]
        train_dataset = TensorDataset(train_x["input_ids"], train_x["attention_mask"], train_Y)

        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def getTorchValidationData(self) -> DataLoader:
        torch.manual_seed(self.RANDOM_SEED)
        torch.cuda.manual_seed(self.RANDOM_SEED)

        val_data = pd.read_csv("data/val.txt", sep="\t", names=["dialogue", "response"], header=0)

        val_x = self.tokenizer.batch_encode_plus(val_data["dialogue"].to_list(), return_tensors="pt",
                                                 max_length=self.max_len, padding="max_length", truncation=True)
        val_Y = self.tokenizer.batch_encode_plus((val_data["dialogue"] + val_data["response"]).to_list(),
                                                 return_tensors="pt", max_length=self.max_len, padding="max_length", truncation=True)["input_ids"]
        val_dataset = TensorDataset(val_x["input_ids"], val_x["attention_mask"], val_Y)

        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

    def encoding(self, text: str) -> tf.Tensor:
        # return self.tokenizer.encode(text, return_tensors="tf")
        return self.tokenizer.encode(text, return_tensors="pt")

    def decoding(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)
