import setuptools
import pandas as pd
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import BertForSequenceClassification, BertTokenizerFast

class LabelClassification(LightningModule):
    def __init__(self):
        super(LabelClassification, self).__init__()
        self.RANDOM_SEED = 7777
        pl.seed_everything(self.RANDOM_SEED)

        self.epochs = 10
        self.batch_size = 16
        self.warmup_ratio = 0.1
        self.learning_rate = 5e-3  # 5e-5
        self.num_labels = 7
        self.input_dim = 55  # train - 55, val - 50

        self.MODEL_NAME = "Huffon/klue-roberta-base-nli"
        self.model = BertForSequenceClassification.from_pretrained(self.MODEL_NAME, num_labels=self.num_labels)
        self.tokenizer = BertTokenizerFast.from_pretrained(self.MODEL_NAME)
        self.pad_token_id = self.tokenizer.pad_token_id

        self.label_dict = {'[HAPPY]': 0, '[PANIC]': 1, '[ANGRY]': 2, '[UNSTABLE]': 3, '[HURT]': 4, '[SAD]': 5, '[NEUTRAL]': 6}
        self.train_set = None  # 3366
        self.val_set = None  # 436

    def configure_optimizers(self):
        optim = AdamW(self.parameters(), lr=self.learning_rate)

        num_train_steps = len(self.train_dataloader()) * self.hparams.max_epochs
        num_warmup_steps = int(num_train_steps * self.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 'name': 'cosine_schedule_with_warmup', 'monitor': 'loss', 'interval': 'step', 'frequency': 1}
        return [optim], [lr_scheduler]

    def configure_callbacks(self):
        check_point = ModelCheckpoint("../models/label_classifier/model_ckp/", monitor="val_loss", mode="min")
        early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=4)
        return [check_point, early_stopping]

    def forward(self, x):
        output = self.model(x).logits
        output = F.softmax(output, dim=1)
        return output

    def cross_entropy_loss(self, output, labels):
        return torch.nn.CrossEntropyLoss(reduction='none')(output, labels)

    def accuracy(self, output, labels) -> float:
        output = torch.argmax(output, dim=1)
        return torch.sum(output == labels) / output.__len__() * 100  # %(Precentage)

    def prepare_data(self):
        raw_train = pd.read_csv("../data/label_classification_dataset/train/train_data1.txt", sep="\t", encoding="949").drop(["R1", "R2", "R3"], axis=1)
        raw_val = pd.read_csv("../data/label_classification_dataset/val/val_data1.txt", sep="\t", encoding="949").drop(["R1", "R2", "R3"], axis=1)

        train_x = []
        train_Y = []
        for _, (s1, s2, s3) in raw_train.iterrows():
            try:
                train_x.append(" ".join(s1.split()[1:]))
                train_Y.append(self.label_dict[s1.split()[0]])
            except KeyError:
                train_x = train_x[:-1]
            if s2 != "NONE":
                try:
                    train_x.append(" ".join(s2.split()[1:]))
                    train_Y.append(self.label_dict[s2.split()[0]])
                except KeyError:
                    train_x = train_x[:-1]
            if s3 != "NONE":
                try:
                    train_x.append(" ".join(s3.split()[1:]))
                    train_Y.append(self.label_dict[s3.split()[0]])
                except KeyError:
                    train_x = train_x[:-1]
        train_x = self.tokenizer.batch_encode_plus(train_x, max_length=self.input_dim, padding="max_length", truncation=True, return_tensors="pt")
        train_Y = torch.LongTensor(train_Y)
        self.train_set = TensorDataset(train_x["input_ids"], train_Y)

        val_x = []
        val_Y = []
        for _, (s1, s2, s3) in raw_val.iterrows():
            try:
                val_x.append(" ".join(s1.split()[1:]))
                val_Y.append(self.label_dict[s1.split()[0]])
            except KeyError:
                val_x = val_x[:-1]
            if s2 != "NONE":
                try:
                    val_x.append(" ".join(s2.split()[1:]))
                    val_Y.append(self.label_dict[s2.split()[0]])
                except KeyError:
                    val_x = val_x[:-1]
            if s3 != "NONE":
                try:
                    val_x.append(" ".join(s3.split()[1:]))
                    val_Y.append(self.label_dict[s3.split()[0]])
                except KeyError:
                    val_x = val_x[:-1]
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

        self.log_dict({'loss': loss, 'acc': accuracy}, prog_bar=True, sync_dist=True)
        return {'loss': loss, 'acc': accuracy}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.cross_entropy_loss(y_pred, y)
        accuracy = self.accuracy(y_pred, y)

        self.log_dict({'val_loss': loss, 'val_acc': accuracy}, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'val_loss': loss, 'val_acc': accuracy}

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([output['val_loss'] for output in outputs]).mean()
        mean_acc = torch.stack([output['val_acc'] for output in outputs]).mean()

        self.log_dict({'avg_val_loss': mean_loss, 'avg_val_acc': mean_acc}, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'avg_val_loss': mean_loss, 'avg_val_acc': mean_acc}


model = LabelClassification()
trainer = Trainer(max_epochs=model.epochs, gpus=torch.cuda.device_count(),
                  logger=TensorBoardLogger("../models/label_classifier/tensorboardLog/"))
trainer.fit(model)
torch.save(model, "../models/label_classifier/torch_model/pytorch_model.bin")
trainer.save_checkpoint("../models/label_classifier/pl_model/pl_model.ptl")
