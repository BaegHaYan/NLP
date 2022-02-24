import setuptools
import re
import torch
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import BertForSequenceClassification, BertTokenizerFast

class LabelClassification(LightningModule):
    def __init__(self):
        super(LabelClassification, self).__init__()
        self.RANDOM_SEED = 7777
        torch.manual_seed(self.RANDOM_SEED)
        torch.cuda.manual_seed(self.RANDOM_SEED)
        pl.seed_everything(self.RANDOM_SEED)

        self.num_labels = 7
        self.learning_rate = 5e-5
        self.batch_size = 32
        self.input_dim = None
        self.pad_token_id = self.tokenizer.pad_token_id

        self.MODEL_NAME = "Huffon/klue-roberta-base-nli"
        self.model = BertForSequenceClassification.from_pretrained(self.MODEL_NAME, num_labels=self.num_labels)
        self.tokenizer = BertTokenizerFast.from_pretrained(self.MODEL_NAME)

        self.train_set = None
        self.val_set = None
        self.test_set = None

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = MultiStepLR(optim, milestones=[], gamma=1.0)
        return [optim], [lr_scheduler]

    def forward(self, x):
        output = self.model(x)
        return output.logit

    def cross_entropy_loss(self, output, labels):
        return torch.nn.CrossEntropyLoss()(output, labels, ignore_index=self.pad_token_id)

    def accuracy(self, output, labels) -> float:
        output = torch.argmax(output, dim=1)
        return torch.sum(output == labels) / output.__len__() * 100  # %(Precentage)

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.cross_entropy_loss(y_pred, y)
        accuracy = self.accuracy(y_pred, y)

        logs = {'train_loss': loss, 'train_acc': accuracy}
        return {'loss': loss, 'accuracy': accuracy, 'log': logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.cross_entropy_loss(y_pred, y)
        accuracy = self.accuracy(y_pred, y)

        logs = {'val_loss': loss, 'val_acc': accuracy}
        return {'loss': loss, 'accuracy': accuracy, 'log': logs}

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([output['val_loss'] for output in outputs]).mean()
        mean_acc = torch.stack([output['val_acc'] for output in outputs]).mean()
        logs = {'val_loss': mean_loss, 'val_acc': mean_acc}
        return {'avg_val_loss': mean_loss, 'avg_val_acc': mean_acc, 'log': logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.cross_entropy_loss(y_pred, y)
        accuracy = self.accuracy(y_pred, y)

        logs = {'test_loss': loss, 'test_acc': accuracy}
        return {'loss': loss, 'accuracy': accuracy, 'log': logs}

    def test_epoch_end(self, outputs):
        mean_loss = torch.stack([output['test_loss'] for output in outputs]).mean()
        mean_acc = torch.stack([output['test_acc'] for output in outputs]).mean()
        logs = {'test_loss': mean_loss, 'test_acc': mean_acc}
        return {'avg_test_loss': mean_loss, 'avg_test_acc': mean_acc, 'log': logs}
