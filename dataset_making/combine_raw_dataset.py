from sklearn.model_selection import train_test_split
from transformers import pipeline
from typing import List
import pandas as pd
import jsonlines
import json
import yaml
import os


class Dataset_combiner:
    def __init__(self):
        self.en_to_ko = pipeline("translation_en_to_ko", model=None)
        self.ru_to_en = pipeline("translation_ru_to_en", model="Helsinki-NLP/opus-mt-ru-en")

        self.columns = ["D1", "R1", "D2", "R2", "D3", "R3", "D4", "R4", "D5", "R5"]
        self.train = pd.DataFrame(columns=self.columns)
        self.val = pd.DataFrame(columns=self.columns)
        self.test = pd.DataFrame(columns=self.columns)

    def combine_raw_datasets(self):
        pass

    def processing_Chatbot_data(self) -> pd.DataFrame:
        raw_data = pd.read_csv("../data/raw_dataset/Chatbot/ChatbotData.csv", names=self.columns[:2]+["label"])
        data = raw_data.iloc[1:].drop(["label"], axis=1)
        return data

    def processing_ConversationJSON_data(self) -> pd.DataFrame:
        data = pd.DataFrame(columns=self.columns)
        raw_data = json.load(open("../data/raw_dataset/Conversation_JSON/conversation.json", "r+", encoding="UTF-8"))

        for dialog in raw_data["conversations"]:
            dialog = self.process_line(dialog)
            data = data.append(pd.Series(dialog, index=self.columns).apply(lambda x: self.translate(x)), ignore_index=True)
        return data

    def processing_conversations_data(self) -> pd.DataFrame:
        data = pd.DataFrame(columns=self.columns)
        raw_data = pd.read_csv("../data/raw_dataset/conversations/dialogs.txt", sep="\t", encoding="UTF-8", names=["S1", "S2"])

        line = []
        before_s2 = None
        for i, (s1, s2) in raw_data.iterrows():
            if before_s2 != s1 and before_s2 is not None:
                line.append(self.translate(before_s2))
                line = self.process_line(line)
                data = data.append(pd.Series(line, index=self.columns), ignore_index=True)
                line = []
            line.append(self.translate(s1))
            before_s2 = s2

        line.append(self.translate(before_s2))
        line = self.process_line(line)
        data = data.append(pd.Series(line, index=self.columns), ignore_index=True)
        return data

    def processing_cornell_movie_data(self) -> pd.DataFrame:
        data = pd.DataFrame(columns=self.columns)
        raw_data = pd.read_csv("../data/raw_dataset/cornell_movie_dialogs_corpus/movie_conversations.txt",
                               names=["charID1", "charID2", "movieID", "conv"], sep="+++$+++", encoding="UTF-8")["conv"]
        line_dict = dict([(line_id, self.translate(line)) for _, (line_id, _, _, _, line) in
                          pd.read_csv("../data/raw_dataset/cornell_movie_dialogs_corpus/movie_lines.txt", sep="+++$+++", encoding="UTF-8")])
        for _, conv_list in raw_data.items():
            line = [line_dict[lineID] for lineID in conv_list]
            line = self.process_line(line)
            data = data.append(pd.Series(line, index=self.columns), ignore_index=True)
        return data

    def processing_Depression_data(self) -> pd.DataFrame:
        data = pd.DataFrame(columns=self.columns)
        raw_data = yaml.safe_load(open("../data/raw_dataset/Depression/depression.yml", "r+", encoding="UTF-8"))['conversations']
        for lines in raw_data:
            question = self.translate(lines[0])
            for answer in lines[1:]:
                data = data.append(pd.Series([question, self.translate(answer)], index=self.columns[:2]), ignore_index=True)
        return data

    def processing_HumanConversation_data(self) -> pd.DataFrame:
        data = pd.DataFrame(columns=self.columns)
        raw_data = open("../data/raw_dataset/HumanConversation/human_chat.txt", encoding="UTF-8").readlines()
        line = []
        sep_sent = raw_data[0]
        for sent in raw_data:
            if sent == sep_sent:
                data = data.append(pd.Series(self.process_line(line), index=self.columns), ignore_index=True)
            line.append(self.translate(sent[9:-1]))
        data = data.append(pd.Series(self.process_line(line), index=self.columns), ignore_index=True)
        return data

    def processing_TOQ_Russian_data(self) -> pd.DataFrame:
        data = pd.DataFrame(columns=self.columns)
        raw_data = jsonlines.open("../data/raw_dataset/ThousandsOfQuestions_Russian/data.jsonl")
        for conv in raw_data:
            question = self.translate(conv['question'], is_Ru=True)
            for answer in conv['answers']:
                data = data.append(pd.Series([question, self.translate(answer['text'], is_Ru=True)], index=self.columns[:2]), ignore_index=True)
        return data

    def processing_TopicalChat_data(self) -> pd.DataFrame:
        data = pd.DataFrame(columns=self.columns)
        file_names = os.listdir("../data/raw_dataset/Topical_Chat")
        for file_name in file_names:
            for raw_data in json.load(open("../data/raw_dataset/Topical_Chat/"+file_name, "r+", encoding="UTF-8")):
                lines = []
                for content in raw_data['content']:
                    lines.append(self.translate(content['message']))
                data = data.append(pd.Series(self.process_line(lines), index=self.columns), ignore_index=True)
        return data

    def processing_EmotionalConv_data(self) -> pd.DataFrame:
        data = pd.DataFrame(columns=self.columns)
        file_names = os.listdir("../data/raw_dataset/감성대화")
        is_passing = False
        for file_name in file_names:
            for conv in json.load(open("../data/raw_dataset/감성대화/" + file_name, "r+", encoding="UTF-8")):
                lines = []
                for line in conv['talk']['content'].values():
                    if any(ng_word in line for ng_word in ["아내", "남편", "이혼", "결혼", "노년", "여보"]):
                        is_passing = True
                        break
                    lines.append(line)
                if is_passing:
                    is_passing = False
                    continue
                data = data.append(pd.Series(self.process_line(lines), index=self.columns), ignore_index=True)
        return data

    def process_line(self, line: List) -> List:
        line = line[:10]
        if len(line) % 2 == 1:
            line = line[:-1]
        line = line + [None] * (len(self.columns) - len(line))
        return line

    def translate(self, text: str, is_Ru: bool = False):
        if is_Ru:
            text = self.ru_to_en(text)
        text = self.en_to_ko(text)
        return text[0]["translation_text"]
