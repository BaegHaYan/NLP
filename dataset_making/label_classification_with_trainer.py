from transformers import ElectraForSequenceClassification, ElectraTokenizerFast
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from torch.utils.data import TensorDataset
import pandas as pd
import datetime
import torch
import re
import os

PREMODEL_NAME = "monologg/koelectra-base-v3-discriminator"
RANDOM_SEED = 7777
label_dict = {'[HAPPY]': 0, '[PANIC]': 1, '[ANGRY]': 2, '[UNSTABLE]': 3, '[HURT]': 4, '[SAD]': 5, '[NEUTRAL]': 6}


def getDataset(isTrain: bool, using_device: str):
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)

    if isTrain:
        data_path = "../data/label_classification_dataset/train/train_data1.txt"
    else:
        data_path = "../data/label_classification_dataset/val/val_data1.txt"

    data = pd.read_csv(data_path, sep="\t", encoding="949").drop(["R1", "R2", "R3"], axis=1)

    x = []
    Y = []
    for _, row in data.iterrows():
        for s in row:
            if any(label in s for label in label_dict.keys()):
                s = re.sub('(.+)(\[.*])', r'\2 \1', s)
                s = re.sub('(\[.*])', r'\1 ', s)
                if re.search('(\[.*])', s) is None:
                    break
                x.append(" ".join(s.split()[1:]))
                Y.append(label_dict[s.split()[0]])
    x = tokenizer.batch_encode_plus(x, padding=True, return_tensors="pt")
    Y = torch.LongTensor(Y)
    dataset = TensorDataset(x["input_ids"], Y)
    return dataset


device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 10
batch_size = 16

model = ElectraForSequenceClassification.from_pretrained(PREMODEL_NAME, num_labels=7).to(device)
tokenizer = ElectraTokenizerFast.from_pretrained(PREMODEL_NAME)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

log_dir = os.path.join('../models/label_classifier/trainer/log/', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
train_args = TrainingArguments(output_dir="../models/label_classifier/trainer/output_dir/",
                               logging_dir=log_dir,
                               num_train_epochs=epochs,
                               per_device_train_batch_size=batch_size,
                               per_device_eval_batch_size=batch_size,
                               learning_rate=3e-5,
                               weight_decay=0.01,
                               )

trainer = Trainer(model=model, args=train_args, tokenizer=tokenizer, data_collator=data_collator,
                  train_dataset=getDataset(isTrain=True, using_device=device),
                  eval_dataset=getDataset(isTrain=False, using_device=device))
trainer.train()
torch.save(model, "../models/label_classifier/trainer/pytorch_model.bin")
