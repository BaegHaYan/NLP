from transformers import GPT2TokenizerFast, BertTokenizerFast
from typing import Sequence
import tensorflow as tf
import pandas as pd
import random
import json
import os

def make_RowDataset():
    train = pd.DataFrame()

def make_Dataset():
    pass

class Preprocesser:
    def __init__(self):
        self.RANDOM_SEED = 10
        # HyperParam
        self.batch_size = 16
        self.max_len = None
        # data
        self.data_num = None
        # self.PREMODEL_NAME = "kakaobrain/kogpt"
        # self.REVISION_NAME ='KoGPT6B-ryan1.5b-float16'
        self.PREMODEL_NAME = "byeongal/Ko-DialoGPT"
        self.COMPRESS_MODEL_NAME = "monologg/kobert"
        # tokenizers
        self.tokenizer = GPT2TokenizerFast.from_pretrained("./tokenizer")
        self.compress_tokenizer = BertTokenizerFast.from_pretrained(self.COMPRESS_MODEL_NAME)
        self.vocab_size = self.tokenizer.vocab_size

    def getTrainData(self) -> tf.data.Dataset:
        # data's dialogue : S1</s>S2</s> | response : R1</s>
        trainData = pd.read_csv("data/train.txt", sep="\t", names=["dialogue", "response"])

        train_x = self.tokenizer.batch_encode_plus(trainData["dialogue"].to_list(), return_tensors="tf",
                                                   max_length=self.max_len, padding="max_length", truncation=True)
        encoded_train_x = dict()
        for key, value in train_x.items():
            encoded_train_x[key] = value

        train_Y = self.tokenizer.batch_encode_plus((trainData["dialogue"] + trainData["response"]).to_list(), return_tensors="tf",
                                                   max_length=self.max_len, padding="max_length", truncation=True)["input_ids"]
        return tf.data.Dataset.from_tensor_slices((encoded_train_x, train_Y)).batch(self.batch_size).shuffle(256, seed=self.RANDOM_SEED)

    def getValidationData(self) -> tf.data.Dataset:
        valData = pd.read_csv("data/val.txt", sep="\t", names=["dialogue", "response"])

        val_x = self.tokenizer.batch_encode_plus(valData["dialogue"].to_list(), return_tensors="tf",
                                                 max_length=self.max_len, padding="max_length", truncation=True)
        encoded_val_x = dict()
        for key, value in val_x.items():
            encoded_val_x[key] = value

        val_Y = self.tokenizer.batch_encode_plus((valData["dialogue"] + valData["response"]).to_list(), return_tensors="tf",
                                                 max_length=self.max_len, padding="max_length", truncation=True)["input_ids"]
        return tf.data.Dataset.from_tensor_slices((encoded_val_x, val_Y)).batch(self.batch_size).shuffle(256, seed=self.RANDOM_SEED)

    def encoding(self, text: str) -> tf.Tensor:
        return self.tokenizer.encode(text, return_tensors="tf")

    def decoding(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)




