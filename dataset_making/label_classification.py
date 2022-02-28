import pandas as pd
import setuptools
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
        self.input_dim = None  # train - 55
        self.pad_token_id = self.tokenizer.pad_token_id

        self.MODEL_NAME = "Huffon/klue-roberta-base-nli"
        self.model = BertForSequenceClassification.from_pretrained(self.MODEL_NAME, num_labels=self.num_labels)
        self.tokenizer = BertTokenizerFast.from_pretrained(self.MODEL_NAME)

        self.label_dict = {'[HAPPY]': 0, '[PANIC]': 1, '[ANGRY]': 2, '[UNSTABLE]': 3, '[HURT]': 4, '[SAD]': 5, '[NEUTRAL]': 6}
        self.train_set = None  # 3366
        self.val_set = None

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
        raw_train = pd.read_csv("../data/encoded_dataset/train/train_data1.txt", sep="\t", encoding="949").drop(["R1", "R2", "R3"], axis=1)
        raw_val = pd.read_csv("../data/encoded_dataset/val/val_data1.txt", sep="\t", encoding="949").drop(["R1", "R2", "R3"], axis=1)

        train_x = []
        train_Y = []
        for _, (s1, s2, s3) in raw_train.iterrows():
            train_x.append(" ".join(s1.split()[1:]))
            train_Y.append(self.label_dict[s1.split()[0]])
            if s2 != "NONE":
                train_x.append(" ".join(s2.split()[1:]))
                train_Y.append(self.label_dict[s2.split()[0]])
            if s3 != "NONE":
                train_x.append(" ".join(s3.split()[1:]))
                train_Y.append(self.label_dict[s3.split()[0]])
        train_x = self.tokenizer.batch_encode_plus(train_x, max_length=self.input_dim, padding="max_length", truncation=True, return_tensors="pt")
        train_Y = torch.LongTensor(train_Y)
        self.train_set = TensorDataset(train_x["input_ids"], train_Y)

        val_x = []
        val_Y = []
        for _, (s1, s2, s3) in raw_val.iterrows():
            val_x.append(" ".join(s1.split()[1:]))
            val_Y.append(self.label_dict[s1.split()[0]])
            if s2 != "NONE":
                val_x.append(" ".join(s2.split()[1:]))
                val_Y.append(self.label_dict[s2.split()[0]])
            if s3 != "NONE":
                val_x.append(" ".join(s3.split()[1:]))
                val_Y.append(self.label_dict[s3.split()[0]])
        val_x = self.tokenizer.batch_encode_plus(val_x, max_length=self.input_dim, padding="max_length", truncation=True, return_tensors="pt")
        val_Y = torch.LongTensor(val_Y)
        self.val_set = TensorDataset(val_x["input_ids"], val_Y)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True)

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
