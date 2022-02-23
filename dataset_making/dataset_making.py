from transformers import GPT2TokenizerFast
import pandas as pd
import random
import json
import os

def make_Dataset():
    tokenizer = GPT2TokenizerFast.from_pretrained("tokenizer")

    train = pd.DataFrame(columns=["dialogue", "response"])
    for file_name in os.listdir("data/encoded_dataset/train"):
        data = pd.read_csv("../data/encoded_dataset/train/"+file_name, sep="\t", encoding="949", header=0)

        for _, conv in data.iterrows():
            temp_d = ""
            for i, (key, value) in enumerate(conv.items()):
                if i == 5 or conv.iloc[i+1] == "NONE":
                    train = train.append(pd.DataFrame([[temp_d, value + tokenizer.eos_token]], columns=["dialogue", "response"]))
                    break
                if key[0] == "R":
                    train = train.append(pd.DataFrame([[temp_d, value + tokenizer.eos_token]], columns=["dialogue", "response"]))

                temp_d += value.strip() + tokenizer.eos_token
    train.to_csv("../data/train.txt", sep="\t", encoding="utf-8", index=False)

    val = pd.DataFrame(columns=["dialogue", "response"])
    for file_name in os.listdir("data/encoded_dataset/val"):
        data = pd.read_csv("../data/encoded_dataset/val/"+file_name, sep="\t", encoding="949", header=0)

        for _, conv in data.iterrows():
            temp_d = ""
            for i, (key, value) in enumerate(conv.items()):
                if i == 5 or conv.iloc[i + 1] == "NONE":
                    val = val.append(
                        pd.DataFrame([[temp_d, value + tokenizer.eos_token]], columns=["dialogue", "response"]))
                    break
                if key[0] == "R":
                    val = val.append(
                        pd.DataFrame([[temp_d, value + tokenizer.eos_token]], columns=["dialogue", "response"]))

                temp_d += value.strip() + tokenizer.bos_token
    val.to_csv("../data/val.txt", sep="\t", encoding="utf-8", index=False)
