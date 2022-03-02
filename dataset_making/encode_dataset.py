import pandas as pd
import torch
import os
import re
# TODO 모델 import

class Dataset_encoder:
    def __init__(self):
        self.persona_changer = None
        self.data_path = "../data/combined_dataset/"

    def encoding_dataset(self):
        for file_name in os.listdir(self.data_path):
            data = pd.read_csv(self.data_path+file_name, sep="\t", encoding="UTF-8")
            for col in data.columns:
                data[col] = data[col].apply(lambda x: re.sub("[^가-힣0-9a-zA-z,.?! ]", "", x))
                if "R" in col:
                    data[col] = data[col].apply(lambda x: self.change_persona(x))
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

    def change_persona(self, data):
        return self.persona_changer(data)


