import setuptools
from transformers import ElectraForSequenceClassification, BartForConditionalGeneration
from transformers import ElectraTokenizerFast, BartTokenizerFast
import pandas as pd
import logging
import torch
import os
import re

class Dataset_encoder:
    def __init__(self):
        self.label_dict = {0: '[HAPPY]', 1: '[PANIC]', 2: '[ANGRY]', 3: '[UNSTABLE]', 4: '[HURT]', 5: '[SAD]', 6: '[NEUTRAL]'}
        self.label_classifier = ElectraForSequenceClassification.from_pretrained("../models/label_classifier/trainer")
        self.label_tokenizer = ElectraTokenizerFast.from_pretrained("../tokenizer/GPT")
        self.persona_convertor = BartForConditionalGeneration.from_pretrained("../models/persona_converter/trainer")
        self.persona_tokenizer = BartTokenizerFast.from_pretrained("../tokenizer/koBart")
        self.data_path = "../data/combined_dataset/"

    def encoding_dataset(self):
        for file_name in os.listdir(self.data_path):
            data = pd.read_csv(self.data_path+file_name, sep="\t", encoding="UTF-8")
            for col in data.columns:
                data[col] = data[col].apply(lambda x: re.sub("[^가-힣0-9a-zA-z,.?! ]", "", x))
                if "R" in col:
                    data[col] = data[col].apply(lambda x: self.change_persona(x))
                else:
                    data[col] = data[col].apply(lambda x: self.label_classification(x))
            encoded_data = pd.DataFrame(columns=["Q", "A"])
            for _, row in data.iterrows():
                q_list = []
                for idx in row.index:
                    if row[idx] == "None":
                        break
                    if "R" in idx:
                        encoded_data = encoded_data.append(pd.Series(["</s>".join(q_list)+"</s>", row[idx]], index=["Q", "A"]), ignore_index=True)
                    q_list.append(row[idx])
            data.to_csv("../data/"+file_name, sep="\t", encoding="UTF-8")

    def label_classification(self, data: str) -> str:
        inputs = self.label_tokenizer(data, return_tensors="pt")["input_ids"]
        output = self.label_classifier(inputs).logits
        output = torch.argmax(output, dim=-1).item()

        return self.label_dict[output] + " " + data

    def change_persona(self, data: str) -> str:
        data = self.persona_tokenizer(data, return_tensors="pt")["input_ids"]
        output = self.persona_convertor(data).logits
        output = torch.argmax(output, dim=-1)
        output_text = self.persona_tokenizer.decode(output[0], skip_special_tokens=True)
        return output_text


def getInvalidPersonaData():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt=None, style='$'))
    logger.addHandler(handler)

    train = pd.read_csv("../data/label_classification_dataset/train/train_data1.txt", sep="\t", encoding="949")
    val = pd.read_csv("../data/label_classification_dataset/val/val_data1.txt", sep="\t", encoding="949")
    data = train.append(val)
    invalidPersona_list = ["니다", "세요", "였다", "요.", "요?", "요!"]

    invalid_data = []
    for row in ["R1", "R2", "R3"]:
        for line in data[row].values:
            line = line.strip()
            if re.match("NONE", line) is not None:
                continue
            if any(w_ward in line for w_ward in invalidPersona_list):
                invalid_data.append(line)
    if len(invalid_data) < 200:
        wrongPersona_list = invalidPersona_list + ["여보", "부인", "자식", "남편", "결혼", "이혼", "노후"]
        for row in ["S1", "S2", "S3"]:
            for line in data[row].values:
                line = line.strip()
                if len(invalid_data) > 200:
                    break
                if re.match("NONE", line) is not None:
                    continue
                if any(w_ward in line for w_ward in wrongPersona_list):
                    invalid_data.append(line)
    invalid_data.extend(data["S1"].tail().to_list())
    logger.info(f"total num of data : {len(invalid_data)}")
    pd.Series(invalid_data).to_csv("../data/persona_dataset/source_data.txt", sep="\t", encoding="UTF-8")
