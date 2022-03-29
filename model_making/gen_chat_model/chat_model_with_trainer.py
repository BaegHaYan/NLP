from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from typing import List, Dict
import pandas as pd
import datetime
import torch
import os

if __name__ == "__main__":
    PREMODEL_NAME = "byeongal/Ko-DialoGPT"
    RANDOM_SEED = 7777
    MAX_LEN = None

    def getDataset(data_type: str, tokenizer, using_device: str) -> List[Dict[str, torch.Tensor]]:
        torch.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed(RANDOM_SEED)
        src_max_len = None
        trg_max_len = None

        if data_type == "train":
            data_path = "../../data/train.txt"
        elif data_type == "val":
            data_path = "../../data/val.txt"
        elif data_type == "test":
            data_path = "../../data/test.txt"
        else:
            raise ValueError("data type must to be one of the 'train'/'val'/'test'.")
        encoded_data = []
        data = pd.read_csv(data_path, sep="\t", encoding="utf-8", index_col=0)
        q = tokenizer.batch_encode_plus(data["Q"].to_list(), max_length=src_max_len, padding="max_length",
                                        truncation=True, return_tensors="pt").to(using_device)
        a = tokenizer.batch_encode_plus(data["A"].to_list(), max_length=trg_max_len, padding="max_length",
                                        truncation=True, return_tensors="pt")["input_ids"].to(using_device)
        for q_ids, q_att, q_type, a in zip(q["input_ids"], q["attention_mask"], q["token_type_ids"], a):
            encoded_data.append({"input_ids": q_ids, "attention_mask": q_att, "token_type_ids": q_type,
                                 "decoder_input_ids": "<s>" + a, "labels": a + "</s>"})
        return encoded_data

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 10
    batch_size = 16

    model = GPT2LMHeadModel.from_pretrained(PREMODEL_NAME).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained("../../tokenizer/GPT")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    log_dir = os.path.join('../../models/chat_model/trainer/logs', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    train_args = TrainingArguments(output_dir="../model/torch_models/traniner_model",
                                   logging_dir=log_dir,
                                   num_train_epochs=epochs,
                                   per_device_train_batch_size=batch_size,
                                   per_device_eval_batch_size=batch_size,
                                   learning_rate=3e-5,
                                   weight_decay=0.01,
                                   )

    trainer = Trainer(model=model, args=train_args, tokenizer=tokenizer, data_collator=data_collator,
                      train_dataset=getDataset("train", tokenizer=tokenizer, using_device=device),
                      eval_dataset=getDataset("val", tokenizer=tokenizer, using_device=device))
    trainer.train()
