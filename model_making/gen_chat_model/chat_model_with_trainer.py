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

    def getDataset(isTrain: bool, using_device: str) -> List[Dict[str, torch.Tensor]]:  # TODO
        torch.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed(RANDOM_SEED)
        if isTrain:
            data_path = None
        else:
            data_path = None
        encoded_data = []
        pass
        return encoded_data

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 5
    batch_size = 16

    model = GPT2LMHeadModel.from_pretrained(PREMODEL_NAME).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained("../../tokenizer")
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
                      train_dataset=getDataset(isTrain=True, using_device=device), eval_dataset=getDataset(isTrain=False, using_device=device))
    trainer.train()
