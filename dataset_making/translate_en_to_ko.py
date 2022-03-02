import setuptools
from transformers import MBartForConditionalGeneration, MBartTokenizerFast
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
import pandas as pd
import torch
import os
import re

class Translater_en_to_ko(pl.LightningModule):
    def __init__(self, epochs: int, gamma: float = 0.5):
        super(Translater_en_to_ko, self).__init__()
        self.RANDOM_SEED = 7777
        torch.manual_seed(self.RANDOM_SEED)
        torch.cuda.manual_seed(self.RANDOM_SEED)
        pl.seed_everything(self.RANDOM_SEED)

        self.model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")
        self.tokenizer = MBartTokenizerFast.from_pretrained("facebook/mbart-large-cc25")

        self.epochs = epochs
        self.gamma = gamma
        self.batch_size = 32
        self.max_len_x = 82
        self.max_len_y = 78

        self.train_set = None
        self.val_set = None

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = MultiStepLR(optim, milestones=[int(self.epochs*0.3), int(self.epochs*0.6)], gamma=self.gamma)
        return [optim], [lr_scheduler]

    def forward(self, x):
        output = self.model(x)
        return output.logit

    def cross_entropy_loss(self, output, labels):
        return torch.nn.CrossEntropyLoss()(output, labels, ignore_index=self.pad_token_id)

    def prepare_data(self):
        data = pd.DataFrame(columns=['번역문', '원문'])
        for file_name in os.listdir("../data/translate_en_to_ko_dataset"):
            data = data.append(pd.read_excel("../data/translate_en_to_ko_dataset/"+file_name), ignore_index=True)
        for col in data.columns:
            if any(col == c for c in ['번역문', '원문']):
                continue
            data.drop([col], axis=1, inplace=True)
        data.columns = ['en', 'ko']

        data['ko'] = data['ko'].apply(lambda x: self.change_korean_honorific(x))
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

        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.cross_entropy_loss(y_pred, y)

        logs = {'val_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([output['val_loss'] for output in outputs]).mean()
        logs = {'val_loss': mean_loss}
        return {'avg_val_loss': mean_loss, 'log': logs}

    def change_korean_honorific(self, sent) -> str:
        sent = re.sub("그렇습니다", "그래", sent)
        sent = re.sub("주세요\.", "줄래?", sent)
        sent = re.sub("하세요", "해", sent)
        sent = re.sub("합니다", "해", sent)
        sent = re.sub("하였다", "했어", sent)
        sent = re.sub("습니다", "어", sent)
        sent = re.sub("제가", "내가", sent)
        sent = re.sub("제게", "내게", sent)
        sent = re.sub("저희", "우리", sent)
        sent = re.sub("요", "", sent)
        sent = re.sub("네([,.?!])", r"응\1", sent)
        return sent


epochs = 5
model = Translater_en_to_ko(epochs)
trainer = pl.Trainer(max_epochs=epochs, gpus=torch.cuda.device_count(),
                     callbacks=[ModelCheckpoint("../models/label_classifier/model_ckp/", verbose=True, monitor="val_acc", mode="max"),
                                EarlyStopping(monitor="val_loss", mode="min", patience=3)])

trainer.fit(model)
torch.save(model.state_dict(), "../models/label_classifier/torch_model/model_state.pt")
trainer.save_checkpoint("../models/label_classifier/pl_model/pytorch_model.bin")
