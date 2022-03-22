import transformers.integrations
from transformers import ElectraForSequenceClassification, ElectraTokenizerFast
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from transformers.integrations import TensorBoardCallback
from transformers import PrinterCallback
import pandas as pd
import datetime
import logging
import torch
import re
import os

MAX_LEN = 55
PREMODEL_NAME = "monologg/koelectra-base-v3-discriminator"
RANDOM_SEED = 7777
label_dict = {'[HAPPY]': 0, '[PANIC]': 1, '[ANGRY]': 2, '[UNSTABLE]': 3, '[HURT]': 4, '[SAD]': 5, '[NEUTRAL]': 6}

def accuracy(pred):
    labels = pred.label_ids
    output = pred.predictions

    output = torch.argmax(torch.LongTensor(output), dim=1)
    output = torch.sum(output == labels) / output.__len__() * 100  # %(Precentage)
    return {'accuracy': output}

def getDataset(isTrain: bool, using_device: str):
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)

    if isTrain:
        data_path = "../data/label_classification_dataset/train/train_data1.txt"
    else:
        data_path = "../data/label_classification_dataset/val/val_data1.txt"

    encoded_data = []
    data = pd.read_csv(data_path, sep="\t", encoding="949").drop(["R1", "R2", "R3"], axis=1)
    for _, row in data.iterrows():
        for s in row:
            if any(label in s for label in label_dict.keys()):
                s = re.sub('(.+)(\[.*])', r'\2 \1', s)
                s = re.sub('(\[.*])', r'\1 ', s)
                if re.search('(\[.*])', s) is None:
                    break

                temp_data = dict()
                x = tokenizer(" ".join(s.split()[1:]), return_tensors="pt", max_length=MAX_LEN, padding="max_length", truncation=True)
                for k, v in x.items():
                    temp_data[k] = v[0].to(using_device)
                temp_data['labels'] = label_dict[s.split()[0]]
                encoded_data.append(temp_data)
    return encoded_data


device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 10
batch_size = 16

model = ElectraForSequenceClassification.from_pretrained(PREMODEL_NAME, num_labels=7).to(device)
tokenizer = ElectraTokenizerFast.from_pretrained(PREMODEL_NAME)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

log_dir = os.path.join('../models/label_classifier/trainer/log/', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
train_args = TrainingArguments(output_dir="../models/label_classifier/trainer/output_dir/",
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
                               save_strategy="epoch",
                               )

trainer = Trainer(model=model, args=train_args, data_collator=data_collator, compute_metrics=accuracy,
                  callbacks=[PrinterCallback(), TensorBoardCallback()],
                  train_dataset=getDataset(isTrain=True, using_device=device),
                  eval_dataset=getDataset(isTrain=False, using_device=device))
trainer.train()
torch.save(model, "../models/label_classifier/trainer/pytorch_model.bin")
