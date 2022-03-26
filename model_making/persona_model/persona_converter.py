from torch.utils import mobile_optimizer
from transformers import BartForConditionalGeneration, BartTokenizerFast
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from transformers.integrations import TensorBoardCallback
from transformers import PrinterCallback
import pandas as pd
import logging
import datetime
import torch
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter())
logger.addHandler(handler)

def getDataset(isTrain: bool, using_device: str):
    # [{key: value}]
    RANDOM_SEED = 7777
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    scr_input_dim = 37
    trg_input_dim = 23

    if isTrain:
        data_path = "../../data/persona_dataset/data1.txt"
        repeat_num = 10
    else:
        data_path = "../../data/persona_dataset/data1.txt"
        repeat_num = 1

    encoded_data = []
    for _ in range(repeat_num):
        data = pd.read_csv(data_path, sep="\t", encoding="949")
        data.dropna(inplace=True)
        for _, (non_persona, persona) in data.iterrows():
            temp_data = tokenizer(tokenizer.cls_token + non_persona, max_length=scr_input_dim,
                                  padding="max_length", truncation=True, return_tensors="pt").to(using_device)
            encoded_dict = dict()
            for k, v in temp_data.items():
                encoded_dict[k] = v[0].to(using_device)

            encoded_dict["decoder_input_ids"] = tokenizer.encode(tokenizer.cls_token + persona, max_length=trg_input_dim,
                                                                 padding="max_length", truncation=True, return_tensors="pt")
            encoded_dict["labels"] = tokenizer.encode(persona + tokenizer.sep_token, max_length=trg_input_dim,
                                                      padding="max_length", truncation=True, return_tensors="pt")
            encoded_data.append(encoded_dict)
    return encoded_data


device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 10
batch_size = 32
PREMODEL_NAME = "hyunwoongko/kobart"

model = BartForConditionalGeneration.from_pretrained(PREMODEL_NAME, num_labels=7).to(device)
tokenizer = BartTokenizerFast.from_pretrained(PREMODEL_NAME)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

log_dir = os.path.join('../../models/persona_converter/trainer/logs/', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
train_args = TrainingArguments(output_dir="../../models/persona_converter/trainer/output_dir/",
                               logging_dir=log_dir,
                               do_train=True,
                               do_eval=True,
                               learning_rate=3e-5,
                               per_device_train_batch_size=batch_size,
                               per_device_eval_batch_size=batch_size,
                               num_train_epochs=epochs,
                               weight_decay=0.01,
                               load_best_model_at_end=True,
                               evaluation_strategy="epoch",
                               # save_strategy="epoch",  # it makes error in pyCharm, but it prevents error in colab
                               )

trainer = Trainer(model=model, args=train_args, data_collator=data_collator,
                  callbacks=[PrinterCallback(), TensorBoardCallback()],
                  train_dataset=getDataset(isTrain=True, using_device=device),
                  eval_dataset=getDataset(isTrain=False, using_device=device))
trainer.train()
torch.save(model, "../../models/persona_converter/trainer/pytorch_model.bin")
