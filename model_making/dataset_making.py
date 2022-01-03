from transformers import GPT2TokenizerFast
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

    def check_dataset(order_dataset: str):
        assert order_dataset == "train" or order_dataset == "val"
        if order_dataset == "train":
            nonlocal train
            if train.iloc[-1].name >= 50000:
                nonlocal train_file_num
                train.to_csv(f"data/combined_raw_dataset/{order_dataset}/raw_data{train_file_num}.tsv", sep="\t", encoding="utf-8", index=False, na_rep="NONE")
                print(f"Train file raw_data{train_file_num} was saved.")
                train_file_num += 1
                train = pd.DataFrame(columns=columns)
        else:
            nonlocal val
            if val.iloc[-1].name >= 50000:
                nonlocal val_file_num
                val.to_csv(f"data/combined_raw_dataset/{order_dataset}/raw_data{val_file_num}.tsv", sep="\t", encoding="utf-8", index=False, na_rep="NONE")
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
        print(emotional_file_name + " file start.")
        for conv in json.load(open("data/raw_dataset/감성대화/"+emotional_file_name, "r+", encoding="utf-8")):
            temp = []
            for sent in conv["talk"]["content"].values():
                temp.append(sent)

            if "Training" in emotional_file_name:
                train = train.append(pd.DataFrame([temp], columns=["S1", "R1", "S2", "R2", "S3", "R3"]), ignore_index=True)
                check_dataset("train")
            else:
                val = val.append(pd.DataFrame([temp], columns=["S1", "R1", "S2", "R2", "S3", "R3"]), ignore_index=True)
                check_dataset("val")
    print("EmotionalData ended.")

    # kcs
    for order in ["Training", "Validation"]:
        print("Korean Conversation Summary " + order + " start.")
        for filename in os.listdir("./data/raw_dataset/한국어 대화 요약/"+order):
            kcs_data = json.load(open("./data/raw_dataset/한국어 대화 요약/"+order+"/"+filename, "r+", encoding="utf-8"))["data"]
            print(order+" "+filename+" file start.")
            for conv in kcs_data:
                turns = conv["header"]["dialogueInfo"]["numberOfTurns"]
                if conv["header"]["dialogueInfo"]["numberOfParticipants"] != 2:
                    continue
                if turns % 2 == 1 or turns > 6:
                    continue

                temp = []
                temp_sent = ""
                pre_turnID = ""
                for dialogue in conv["body"]["dialogue"]:
                    if dialogue["turnID"] != pre_turnID and pre_turnID != "":
                        temp.append(temp_sent)
                        temp_sent = ""

                    temp_sent += dialogue["utterance"] + " "
                    pre_turnID = dialogue["turnID"]
                temp.append(temp_sent)

                temp_col = []
                for i in range(int(turns/2)):
                    temp_col.append(f"S{i+1}")
                    temp_col.append(f"R{i+1}")

                if order == "Training":
                    train = train.append(pd.DataFrame([temp], columns=temp_col), ignore_index=True)
                    check_dataset("train")
                else:
                    val = val.append(pd.DataFrame([temp], columns=temp_col), ignore_index=True)
                    check_dataset("val")
    print("KoreanConversationSummary ended.")

    # korean SNS
    for filename in os.listdir("data/raw_dataset/한국어_SNS"):
        try:
            data = json.load(open("data/raw_dataset/한국어_SNS/"+filename, "r+", encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        print("start Korean SNS " + filename + "file.")
        for conv in data["data"]:
            turns = conv["header"]["dialogueInfo"]["numberOfTurns"]
            if conv["header"]["dialogueInfo"]["numberOfParticipants"] != 2:
                continue
            if turns % 2 == 1 or turns > 6:
                continue

            temp = []
            temp_sent = ""
            pre_turnID = ""
            for dialogue in conv["body"]:
                if dialogue["turnID"] != pre_turnID and pre_turnID != "":
                    temp.append(temp_sent)
                    temp_sent = ""

                temp_sent += dialogue["utterance"] + " "
                pre_turnID = dialogue["turnID"]
            temp.append(temp_sent)

            temp_col = []
            for i in range(int(turns / 2)):
                temp_col.append(f"S{i + 1}")
                temp_col.append(f"R{i + 1}")

            if random.randint(1, 10) > 3:
                train = train.append(pd.DataFrame([temp], columns=temp_col), ignore_index=True)
                check_dataset("train")
            else:
                val = val.append(pd.DataFrame([temp], columns=temp_col), ignore_index=True)
                check_dataset("val")

    # multi modal
    hist = [""]
    print('start multimodal video dataset')
    for fpath in os.listdir("data/raw_dataset/멀티모달 영상"):
        for fname in os.listdir("./data/raw_dataset/멀티모달 영상/" + fpath):
            try:
                temp_mm = json.load(open("./data/raw_dataset/멀티모달 영상/" + fpath + "/" + fname + "/" + fname + ".json", 'r+', encoding='utf-8'))
            except UnicodeDecodeError:
                temp_mm = json.load(open("./data/raw_dataset/멀티모달 영상/" + fpath + "/" + fname + "/" + fname + ".json", 'r+', encoding='949'))

            temp = []
            for conv in temp_mm['data'].values():  # repeat for all data in this file
                for person in conv.keys():
                    if 'text' not in conv[person].keys():  # find text data
                        continue
                    if conv[person]['text']['script'] == hist[-1]:  # skip duplicate sentence
                        continue
                    hist.append(conv[person]['text']['script'])
                    temp.append(hist[-1])

            temp_col = []
            temp = temp[:6] if len(temp) > 6 else temp
            for i in range(int(len(temp)/2)):
                temp_col.append(f"S{i+1}")
                temp_col.append(f"R{i+1}")
            temp = temp[: len(temp_col)]

            if random.randint(1, 10) > 3:
                train = train.append(pd.DataFrame([temp], columns=temp_col), ignore_index=True)
                check_dataset("train")
            else:
                val = val.append(pd.DataFrame([temp], columns=temp_col), ignore_index=True)
                check_dataset("val")
            print(f"multi_modal {fname[5:]} ended")

    train.to_csv(f"data/combined_raw_dataset/train/raw_data{train_file_num}.tsv", sep="\t", encoding="utf-8", index=False, na_rep="NONE")
    val.to_csv(f"data/combined_raw_dataset/val/raw_data{val_file_num}.tsv", sep="\t", encoding="utf-8", index=False, na_rep="NONE")
    print("making dataset finished.")

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
