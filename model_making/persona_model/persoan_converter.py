import setuptools
import torch
import logging
import argparse
import pandas as pd
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

parser = argparse.ArgumentParser()
parser.add_argument('-e', default=10, type=int, help="epochs", dest="epochs")
parser.add_argument('-b', default=16, type=int, help="batch_size", dest="batch_size")
parser.add_argument('-l', default=5e-5, type=float, help="learning_rate", dest="learning_rate")
parser.add_argument('-g', default=0.9, type=float, help="gamma", dest="gamma")

class PersonaConverter(LightningModule):
    def __init__(self, hparams):
        super(PersonaConverter, self).__init__()
        self.RANDOM_SEED = 7777
        pl.seed_everything(self.RANDOM_SEED)

        self.epochs = hparams.epochs
        self.batch_size = hparams.batch_size
        self.learning_rate = hparams.learning_rate
        self.gamma = hparams.gamma
        self.input_dim = None

        self.MODEL_NAME = None
        self.model = None
        self.tokenizer = None
        self.pad_token_id = self.tokenizer.pad_token_id

        self.train_set = None
        self.val_set = None

    def configure_optimizers(self):
        optim = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.1)

        scheduler = ExponentialLR(optim, gamma=self.gamma)
        lr_scheduler = {'scheduler': scheduler, 'name': 'ExponentialLR', 'monitor': 'loss', 'interval': 'step', 'frequency': 1}
        return [optim], [lr_scheduler]

    def configure_callbacks(self):
        check_point = ModelCheckpoint(dirpath="../models/persona_converter/model_ckp/", filename='{epoch:02d}_{loss:.2f}',
                                      verbose=True, save_last=True, monitor='val_loss', mode='min')
        early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=4)
        return [check_point, early_stopping]

    def forward(self, x):
        # TODO check shape of model output
        output = self.model(x)
        return output.logits

    def cross_entropy_loss(self, output, labels):
        return torch.nn.CrossEntropyLoss()(output, labels)

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.cross_entropy_loss(y_pred, y)
        # TODO check shape of y_pred

        self.log('loss', loss, prog_bar=True, sync_dist=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.cross_entropy_loss(y_pred, y)

        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([output['val_loss'] for output in outputs]).mean()
        mean_acc = torch.stack([output['val_acc'] for output in outputs]).mean()

        self.log('avg_val_loss', mean_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'avg_val_loss': mean_loss}


model = PersonaConverter(parser.parse_args())
trainer = Trainer(max_epochs=model.epochs, gpus=torch.cuda.device_count(),
                  logger=TensorBoardLogger("../models/persona_converter/tensorboardLog/"))
trainer.fit(model)
torch.save(model, "../models/persona_converter/torch_model/pytorch_model.bin")
trainer.save_checkpoint("../models/persona_converter/pl_model/pl_model.ptl")
