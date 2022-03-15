import setuptools
import torch
import argparse
import pandas as pd
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import GPT2TokenizerFast

parser = argparse.ArgumentParser()
parser.add_argument("-e", type=int, default=50, dest="epochs", help="num of epochs")
parser.add_argument("-b", type=int, default=32, dest="batch_size", help="size of each batch")
parser.add_argument("-lr", type=float, default=0.1, dest="learning_rate", help="learning rate")
parser.add_argument("-p", type=int, default=5, dest="patience", help="number of check with no improved")
parser.add_argument("-gamma", type=float, default=0.9, dest="gamma", help="decay rate of learning_rate on each epoch(lr*gamma for every epoch)")
parser.add_argument("-repeat-data", type=int, default=3, dest="repeat", help="number of repeating dataset")


class PersonaConverter(LightningModule):
    def __init__(self, hparams):
        super(PersonaConverter, self).__init__()
        self.RANDOM_SEED = 7777
        pl.seed_everything(self.RANDOM_SEED)

        self.epochs = hparams.epochs
        self.batch_size = hparams.batch_size
        self.learning_rate = hparams.learning_rate
        self.patience = hparams.patience
        self.gamma = hparams.gamma
        self.repeat_dataset = hparams.repeat
        self.input_dim = None  # TODO

        self.MODEL_NAME = None
        self.model = None
        self.tokenizer = GPT2TokenizerFast.from_pretrained("../../tokenizer")
        self.pad_token_id = self.tokenizer.pad_token_id

        self.train_set = None
        self.val_set = None

    def configure_optimizers(self):
        optim = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.1)

        scheduler = ExponentialLR(optim, gamma=self.gamma)
        lr_scheduler = {'scheduler': scheduler, 'name': 'ExponentialLR', 'monitor': 'loss', 'interval': 'step', 'frequency': 1}
        return [optim], [lr_scheduler]

    def configure_callbacks(self):
        check_point = ModelCheckpoint(dirpath="../../models/persona_converter/model_ckp/", filename='{epoch:02d}_{loss:.2f}',
                                      verbose=True, save_last=True, monitor='val_loss', mode='min')
        early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=self.patience)
        return [check_point, early_stopping]

    def forward(self, x):
        # TODO check shape of model output
        output = self.model(x)
        return output.logits

    def cross_entropy_loss(self, output, labels):
        return torch.nn.CrossEntropyLoss()(output, labels)

    def prepare_data(self):
        train = pd.read_csv("../../data/persona_dataset/train.txt", sep="\t", encoding="utf-8")
        val = pd.read_csv("../../data/persona_dataset/val.txt", sep="\t", encoding="utf-8")
        self.train_set = TensorDataset(
            self.tokenizer.batch_encode_plus(train['persona'].to_list() * self.repeat_dataset,
                                             padding="max_length", max_length=self.input_dim, return_tensors="pt"),
            self.tokenizer.batch_encode_plus(train['non_persona'].to_list() * self.repeat_dataset,
                                             padding="max_length", max_length=self.input_dim, return_tensors="pt")
        )
        self.val_set = TensorDataset(
            self.tokenizer.batch_encode_plus(val['persona'].to_list() * self.repeat_dataset,
                                             padding="max_length", max_length=self.input_dim, return_tensors="pt"),
            self.tokenizer.batch_encode_plus(val['non_persona'].to_list() * self.repeat_dataset,
                                             padding="max_length", max_length=self.input_dim, return_tensors="pt")
        )

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.cross_entropy_loss(y_pred, y)
        # TODO check shape of y_pred

        self.log('loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.cross_entropy_loss(y_pred, y)

        self.log('val_loss', loss, prog_bar=True, on_step=True)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([output['val_loss'] for output in outputs]).mean()

        self.log('avg_val_loss', mean_loss, on_epoch=True, prog_bar=True)
        return {'avg_val_loss': mean_loss}


model = PersonaConverter(parser.parse_args())
trainer = Trainer(max_epochs=model.epochs, gpus=torch.cuda.device_count(),
                  logger=TensorBoardLogger("../../models/persona_converter/tensorboardLog/"))
trainer.fit(model)
torch.save(model, "../../models/persona_converter/model_state/model_state.pt")
