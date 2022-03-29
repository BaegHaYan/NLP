from transformers import MBartForConditionalGeneration, MBartTokenizerFast, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments, PrinterCallback
from transformers.integrations import TensorBoardCallback
from sklearn.model_selection import train_test_split
import datetime
import pandas as pd
import torch
import os


def prepare_data(using_device):
    scr_input_dim = 0
    trg_input_dim = 0
    tokenizer = MBartTokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="en_XX", tgt_lang="ko_KR")

    data = pd.DataFrame(columns=['번역문', '원문'])
    for file_name in os.listdir("../data/translate_en_to_ko_dataset"):
        data = data.append(pd.read_excel("../data/translate_en_to_ko_dataset/" + file_name), ignore_index=True)
    for col in data.columns:
        if any(col == c for c in ['번역문', '원문']):
            continue
        data.drop([col], axis=1, inplace=True)
    data.columns = ['en', 'ko']

    encoded_data = []
    for _, (en, ko) in data.iterrows():
        encoded_dict = dict()
        temp_data = tokenizer(en, max_length=scr_input_dim, padding="max_length", truncation=True, return_tensors="pt").to(using_device)
        for k, v in temp_data.items():
            encoded_dict[k] = v[0].to(using_device)
        with tokenizer.as_target_tokenizer():
            encoded_dict['labels'] = tokenizer.batch_encode_plus(ko, max_length=trg_input_dim, padding="max_length", truncation=True,
                                                                 return_tensors="pt").input_ids.to(using_device)
        encoded_data.append(dict)

    train, val = train_test_split(encoded_data, train_size=0.7, random_state=7777)
    return train, val


device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 10
batch_size = 32
PREMODEL_NAME = "facebook/mbart-large-50"

model = MBartForConditionalGeneration.from_pretrained(PREMODEL_NAME, num_labels=7).to(device)
tokenizer = MBartTokenizerFast.from_pretrained(PREMODEL_NAME, src_lang="en_XX", tgt_lang="ko_KR")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

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

train, val = prepare_data(device)
trainer = Trainer(model=model, args=train_args, data_collator=data_collator,
                  callbacks=[PrinterCallback(), TensorBoardCallback()],
                  train_dataset=train, eval_dataset=val)
trainer.train()
torch.save(model, "../../models/translator_en_to_ko/trainer/pytorch_model.bin")
