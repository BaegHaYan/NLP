from transformers import GPT2TokenizerFast, BertTokenizerFast
from typing import Sequence, Union
import tensorflow as tf
import pandas as pd
import random
import json
import os

def make_RowDataset():
    columns = ["S1", "R1", "S2", "R2", "S3", "R3"]

    train_file_num = 1
    val_file_num = 1
    train = pd.DataFrame(columns=columns)
    val = pd.DataFrame(columns=columns)

    def check_dataset(order: str):
        assert order == "train" or order == "val"
        if order == "train":
            nonlocal train
            if train.iloc[-1].name >= 20000:
                nonlocal train_file_num
                train.to_csv(f"data/combined_raw_dataset/{order}/raw_data{train_file_num}.tsv", sep="\t", encoding="utf-8", index=False, na_rep="")
                print(f"Train file raw_data{train_file_num} was saved.")
                train_file_num += 1
                train = pd.DataFrame(columns=columns)
        else:
            nonlocal val
            if val.iloc[-1].name >= 20000:
                nonlocal val_file_num
                val.to_csv(f"data/combined_raw_dataset/{order}/raw_data{val_file_num}.tsv", sep="\t", encoding="utf-8", index=False, na_rep="")
                print(f"Validation file raw_data{val_file_num} was saved.")
                val_file_num += 1
                val = pd.DataFrame(columns=columns)

    print("Data making start.")
    # ChatbotData
    chatbot_data = pd.read_csv("data/raw_dataset/Chatbot/ChatbotData.csv", names=["S1", "R1", "labels"]).loc[1:].drop(["labels"], axis=1)
    train = train.append(chatbot_data, ignore_index=True)
    print("Chatbot data ended.")

    # emotional
    for emotional_file_name in os.listdir("data/raw_dataset/감성대화"):
        for conv in json.load(open("data/raw_dataset/감성대화/"+emotional_file_name, "r+", encoding="utf-8")):
            temp = []
            for sent in conv["talk"]["content"].values():
                temp.append(sent)

            if "Training" in emotional_file_name:
                train.append(pd.DataFrame([temp], columns=["S1", "R1", "S2", "R2", "S3", "R3"]), ignore_index=True)
                check_dataset("train")
            else:
                val.append(pd.DataFrame([temp], columns=["S1", "R1", "S2", "R2", "S3", "R3"]), ignore_index=True)
                check_dataset("val")
    print("EmotionalData ended.")



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




