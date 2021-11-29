from transformers import GPT2TokenizerFast
import tensorflow as tf
import pandas as pd
import random
import json
import os

class Preprocesser:
    def __init__(self):
        self.RANDOM_SEED = 10
        # HyperParam
        self.lr = 3e-5
        self.batch_size = 16
        self.input_dim = None
        self.output_dim = None
        # data
        self.data_num = None
        self.PREMODEL_NAME = "byeongal/Ko-DialoGPT"
        # tokenizers
        self.tokenizer = GPT2TokenizerFast.from_pretrained(self.PREMODEL_NAME)

    def getTrainData(self):
        pass

    def getValidationData(self):
        pass

    def encoding(self, text: str):
        return self.tokenizer.encode(text, return_tensors="tf")

    def decoding(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)




