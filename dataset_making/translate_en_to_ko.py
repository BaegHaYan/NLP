from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers import Trainer, TrainingArguments, PrinterCallback
from transformers.integrations import TensorBoardCallback
from sklearn.model_selection import train_test_split
import pandas as pd
import datetime
import logging
import torch
import os

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
PREMODEL_NAME = "facebook/mbart-large-cc25"
TOKENIZER = MBart50TokenizerFast.from_pretrained(PREMODEL_NAME, src_lang="en_XX", tgt_lang="ko_KR")

def prepare_data(using_device):
    scr_input_dim = 80  # 78
    trg_input_dim = 85  # 82
    logger.info("starting prepare data")
    data = pd.read_csv("../data/translate_en_to_ko_dataset/data.txt", sep="\t", encoding="utf-8", index_col=0)
    encoded_data = []
    encoded_en = TOKENIZER.batch_encode_plus(data["en"].to_list(), max_length=scr_input_dim, padding="max_length",
                                             truncation=True, return_tensors="pt").to(using_device)
    with TOKENIZER.as_target_tokenizer():
        encoded_ko = TOKENIZER.batch_encode_plus(data["ko"].to_list(), max_length=trg_input_dim, padding="max_length", truncation=True,
                                                 return_tensors="pt")["input_ids"].to(using_device)
    logger.info("finished tokenizing dataset")
    for i, (en_ids, en_attention, ko) in enumerate(zip(encoded_en["input_ids"], encoded_en["attention_mask"], encoded_ko)):
        encoded_data.append({"input_ids": en_ids, "attention_mask": en_attention, "labels": ko})
    logger.info("finished encoding datasets")
    train, val = train_test_split(encoded_data, train_size=0.7, random_state=7777)
    return train, val


device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 10
batch_size = 16

train, val = prepare_data(device)
model = MBartForConditionalGeneration.from_pretrained(PREMODEL_NAME, num_labels=7).to(device)

log_dir = os.path.join('../../models/translator_en_to_ko/trainer/logs/', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
train_args = TrainingArguments(output_dir="../../models/translator_en_to_ko/trainer/output_dir/",
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

trainer = Trainer(model=model, args=train_args,
                  callbacks=[PrinterCallback(), TensorBoardCallback()],
                  train_dataset=train, eval_dataset=val)
trainer.train()
torch.save(model, "../../models/translator_en_to_ko/trainer/pytorch_model.bin")
