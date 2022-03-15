import setuptools
import argparse
from transformers import MBartForConditionalGeneration, MBartTokenizerFast
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSeq2SeqLM
import pytorch_lightning as pl
import pandas as pd
import torch
import os
import re

parser = argparse.ArgumentParser()
parser.add_argument("-e", type=int, default=5, dest="epochs", help="epochs")
parser.add_argument("-g", type=float, default=0.5, dest="gamma", help="gamma")
parser.add_argument("-lr", type=float, default=5e-5, dest="learning_rate", help="learning_rate")
parser.add_argument("-b", type=int, default=32, dest="batch_size", help="batch_size")
parser.add_argument("-p", type=int, default=5, dest="patience", help="patience")

class Translater_en_to_ko(pl.LightningModule):
    def __init__(self, hparam):
        super(Translater_en_to_ko, self).__init__()
        self.RANDOM_SEED = 7777
        pl.seed_everything(self.RANDOM_SEED)

        self.model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")
        self.persona_changer = AutoModelForSeq2SeqLM.from_pretrained("../models/persona_converter/pytorch_model.bin")
        self.tokenizer = MBartTokenizerFast.from_pretrained("facebook/mbart-large-cc25")
        self.pad_token_id = self.tokenizer.pad_token_id

        self.epochs = hparam.epochs
        self.gamma = hparam.gamma
        self.learning_rate = hparam.learning_rate
        self.batch_size = hparam.batch_size
        self.patience = hparam.patience
        self.src_max_len = 82
        self.trg_max_len = 78

        self.train_set = None
        self.val_set = None

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.1)

        scheduler = ExponentialLR(optim, gamma=self.gamma)
        lr_scheduler = {'scheduler': scheduler, 'name': 'ExponentialLR', 'monitor': 'loss', 'interval': 'step', 'frequency': 1}
        return [optim], [lr_scheduler]

    def configure_callbacks(self):
        check_point = ModelCheckpoint("../models/translater_en_to_ko/model_ckp/", monitor="val_acc", mode="max")
        early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=self.patience)
        return [check_point, early_stopping]

    def forward(self, x):
        output = self.model(x)
        return output.logits

    def cross_entropy_loss(self, output, labels):
        return torch.nn.CrossEntropyLoss(ignore_index=self.pad_token_id)(output, labels)

    def prepare_data(self):
        data = pd.DataFrame(columns=['번역문', '원문'])
        for file_name in os.listdir("../data/translate_en_to_ko_dataset"):
            data = data.append(pd.read_excel("../data/translate_en_to_ko_dataset/"+file_name), ignore_index=True)
        for col in data.columns:
            if any(col == c for c in ['번역문', '원문']):
                continue
            data.drop([col], axis=1, inplace=True)
        data.columns = ['en', 'ko']

        data['ko'] = data['ko'].apply(lambda x: self.change_persona(x))
        data_x = self.tokenizer.batch_encode_plus(data['en'].to_list(), max_length=self.max_len_x, padding="max_length", truncation=True, return_tensors="pt")
        data_Y = self.tokenizer.batch_encode_plus(data['ko'].to_list(), max_length=self.max_len_y, padding="max_length", truncation=True, return_tensors="pt")
        train_x, val_x, train_Y, val_Y = train_test_split(data_x['input_ids'], data_Y['input_ids'], train_size=0.7)

        self.train_set = TensorDataset(train_x, train_Y)
        self.val_set = TensorDataset(val_x, val_Y)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.cross_entropy_loss(y_pred, y)

        self.log('loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.cross_entropy_loss(y_pred, y)

        self.log('val_loss', loss)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([output['val_loss'] for output in outputs]).mean()
        self.log('avg_val_loss', mean_loss)
        return {'avg_val_loss': mean_loss}

    def change_persona(self, sent) -> str:
        sent = self.persona_changer(sent)
        # TODO change Tensor to str
        return sent


model = Translater_en_to_ko(parser.parse_args())
trainer = pl.Trainer(max_epochs=model.epochs, gpus=torch.cuda.device_count())

trainer.fit(model)
torch.save(model, "../models/translater_en_to_ko/torch_model/pytorch_model.bin")
trainer.save_checkpoint("../models/translater_en_to_ko/pl_model/pl_model.ptl")
