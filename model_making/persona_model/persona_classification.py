import setuptools
import torch
import logging
import argparse
import pandas as pd
import pytorch_lightning as pl
from torchmetrics import Accuracy
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import ElectraForSequenceClassification, GPT2TokenizerFast

parser = argparse.ArgumentParser()
parser.add_argument("-e", type=int, default=50, dest="epochs", help="num of epochs")
parser.add_argument("-b", type=int, default=32, dest="batch_size", help="size of each batch")
parser.add_argument("-lr", type=float, default=0.1, dest="learning_rate", help="learning rate")
parser.add_argument("-p", type=int, default=5, dest="patience", help="number of check with no improved")
parser.add_argument("-gamma", type=float, default=0.9, dest="gamma", help="decay rate of learning_rate on each epoch(lr*gamma for every epoch)")
parser.add_argument("-repeat-data", type=int, default=3, dest="repeat", help="number of repeating dataset")


class PersonaClassification(LightningModule):
    def __init__(self, hparams):
        super(PersonaClassification, self).__init__()
        self.RANDOM_SEED = 7777
        pl.seed_everything(self.RANDOM_SEED)

        self.epochs = hparams.epochs
        self.batch_size = hparams.batch_size
        self.learning_rate = hparams.learning_rate
        self.patience = hparams.patience
        self.gamma = hparams.gamma
        self.repeat_dataset = hparams.repeat
        self.input_dim = None

        self.MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
        self.model = ElectraForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self.tokenizer = GPT2TokenizerFast.from_pretrained("../../tokenizer")
        self.pad_token_id = self.tokenizer.pad_token_id
        self.accuracy = Accuracy()

        self.train_set = None
        self.val_set = None

    def configure_optimizers(self):
        optim = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.1)

        scheduler = ExponentialLR(optim, gamma=self.gamma)
        lr_scheduler = {'scheduler': scheduler, 'name': 'ExponentialLR', 'monitor': 'loss', 'interval': 'step', 'frequency': 1}
        return [optim], [lr_scheduler]

    def configure_callbacks(self):
        check_point = ModelCheckpoint(dirpath="../../models/persona_classifier/model_ckp/", filename='{epoch:02d}_{loss:.2f}',
                                      verbose=True, save_last=True, monitor='val_loss', mode='min')
        early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=self.patience)
        return [check_point, early_stopping]

    def forward(self, x):
        # TODO check shape of model output
        output = self.model(x).logits
        output = F.sigmoid(output)
        return output

    def cross_entropy_loss(self, output, labels):
        return torch.nn.BCELoss()(output, labels)

    def prepare_data(self):
        train = pd.read_csv("../../data/persona_dataset/train.txt", sep="\t", encoding="utf-8")
        val = pd.read_csv("../../data/persona_dataset/val.txt", sep="\t", encoding="utf-8")
        data_list = []
        for data in [train, val]:
            x_data = data['persona'].to_list() * self.repeat_dataset
            x_data.extend(data['non_persona'].to_list() * self.repeat_dataset)
            x = self.tokenizer.batch_encode_plus(x_data, padding="max_length", max_length=self.input_dim, return_tensors="pt")
            Y = [0 for _ in range(len(data['persona']))] * self.repeat_dataset
            Y.extend([1 for _ in range(len(data['non_persona']))] * self.repeat_dataset)
            data_list.append((x['input_ids'], torch.LongTensor(Y)))
        self.train_set = TensorDataset(data_list[0][0], data_list[0][1])
        self.val_set = TensorDataset(data_list[1][0], data_list[1][1])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.cross_entropy_loss(y_pred, y)
        # TODO check shape of y_pred
        accuracy = self.accuracy(y_pred, y)

        self.log('loss', loss)
        self.log('acc', accuracy, prog_bar=True)
        return {'loss': loss, 'acc': accuracy}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.cross_entropy_loss(y_pred, y)
        acc = self.accuracy(y_pred, y)

        self.log_dict({'val_loss': loss, 'val_acc': acc}, prog_bar=True)
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([output['val_loss'] for output in outputs]).mean()
        mean_acc = torch.stack([output['val_acc'] for output in outputs]).mean()

        self.log_dict({'avg_val_loss': mean_loss, 'avg_val_acc': mean_acc}, on_epoch=True, prog_bar=True)
        return {'avg_val_loss': mean_loss, 'avg_val_acc': mean_acc}


model = PersonaClassification(parser.parse_args())
trainer = Trainer(max_epochs=model.epochs, gpus=torch.cuda.device_count(),
                  logger=TensorBoardLogger("../../models/persona_classifier/tensorboardLog/"))
trainer.fit(model)
torch.save(model.state_dict(), "../../models/persona_classifier/model_state/model_state.pt")
