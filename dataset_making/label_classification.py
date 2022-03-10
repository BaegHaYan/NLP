import setuptools
import re
import torch
import pandas as pd
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import ElectraTokenizerFast

class LabelClassification(LightningModule):
    def __init__(self, epochs: int, model_name: str, use_nll: bool = False):
        super(LabelClassification, self).__init__()
        self.RANDOM_SEED = 7777
        pl.seed_everything(self.RANDOM_SEED)

        self.epochs = epochs
        self.use_nll = use_nll
        self.model_name = model_name
        self.batch_size = 32
        self.num_labels = 7
        self.input_dim = 55
        self.learning_rate = 0.01
        self.warmup_ratio = 0.05

        self.embedding_size = 512
        self.hidden_size = 256
        self.dropout_rate = 0.1
        self.num_layers = 3

        self.label_dict = {'[HAPPY]': 0, '[PANIC]': 1, '[ANGRY]': 2, '[UNSTABLE]': 3, '[HURT]': 4, '[SAD]': 5, '[NEUTRAL]': 6}
        self.train_set = None  # 3366
        self.val_set = None  # 436
        self.tokenizer = ElectraTokenizerFast.from_pretrained("monologg/koelectra-base-v3-discriminator")
        self.pad_token_id = self.tokenizer.pad_token_id

        self.embedding_layer = torch.nn.Sequential(
            torch.nn.Embedding(self.tokenizer.vocab_size, self.embedding_size, self.pad_token_id),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.ELU(),
            torch.nn.LayerNorm(self.embedding_size, eps=1e-5, elementwise_affine=True)
        )
        if self.model_name == "GRU":
            self.GRU_layer = torch.nn.GRU(self.embedding_size, self.hidden_size, num_layers=self.num_layers,
                                          batch_first=True, dropout=self.dropout_rate)
        elif self.model_name == "LSTM":
            self.LSTM_layer = torch.nn.LSTM(self.embedding_size, self.hidden_size, num_layers=self.num_layers,
                                            batch_first=True, dropout=self.dropout_rate)
        else:
            raise NameError("model_name has to be 'GRU' or 'LSTM'")
        self.rnn_output_layer = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            torch.nn.ELU(),
            torch.nn.LayerNorm(self.hidden_size, eps=1e-5, elementwise_affine=True),
            torch.nn.Dropout(self.dropout_rate)
        )
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim*self.hidden_size, self.num_labels),
            torch.nn.Softmax(dim=1) if not self.use_nll else torch.nn.LogSoftmax(dim=1)
        )

    def configure_optimizers(self):
        optim = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.1)

        num_train_steps = len(self.train_dataloader()) * self.epochs
        num_warmup_steps = int(num_train_steps * self.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 'name': 'ExponentialLR', 'monitor': 'loss', 'interval': 'step', 'frequency': 1}
        return [optim], [lr_scheduler]

    def configure_callbacks(self):
        model_checkpoint = ModelCheckpoint(dirpath=f"../models/label_classifier/{'nll_' if self.use_nll else ''}{self.model_name}_model_ckp/",
                                           filename='{epoch:02d}_{loss:.2f}', verbose=True, save_last=True, monitor='val_loss', mode='min')
        early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=5)
        lr_monitor = LearningRateMonitor(logging_interval="step")
        return [model_checkpoint, early_stopping, lr_monitor]

    def forward(self, x):
        x = self.embedding_layer(x)
        if self.model_name == "GRU":
            h_0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=self.device)
            x, h = self.GRU_layer(x, h_0)
        elif self.model_name == "LSTM":
            h_0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=self.device)
            c_0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size, device=self.device)
            x, (h, c) = self.LSTM_layer(x, (h_0, c_0))
        else:
            raise NameError("model_name has to be 'GRU' or 'LSTM'")
        x = self.rnn_output_layer(x)
        output = self.output_layer(x.view(self.batch_size, self.input_dim*self.hidden_size))
        return output

    def loss(self, output, labels):
        loss_func = torch.nn.CrossEntropyLoss() if not self.use_nll else torch.nn.NLLLoss()
        return loss_func(output, labels)

    def accuracy(self, output, labels):
        output = torch.argmax(output, dim=1)
        return torch.sum(output == labels) / output.__len__() * 100  # %(Precentage)

    def prepare_data(self):
        raw_train = pd.read_csv("../data/label_classification_dataset/train/train_data1.txt", sep="\t", encoding="949").drop(["R1", "R2", "R3"], axis=1)
        raw_val = pd.read_csv("../data/label_classification_dataset/val/val_data1.txt", sep="\t", encoding="949").drop(["R1", "R2", "R3"], axis=1)

        train_x = []
        train_Y = []
        for _, row in raw_train.iterrows():
            for s in row:
                if any(label in s for label in self.label_dict.keys()):
                    s = re.sub('(.+)(\[.*])', r'\2 \1', s)
                    s = re.sub('(\[.*])', r'\1 ', s)
                    if re.search('(\[.*])', s) is None:
                        break
                    train_x.append(" ".join(s.split()[1:]))
                    train_Y.append(self.label_dict[s.split()[0]])
        train_x = self.tokenizer.batch_encode_plus(train_x, max_length=self.input_dim, padding="max_length", truncation=True, return_tensors="pt")
        train_Y = torch.LongTensor(train_Y)
        self.train_set = TensorDataset(train_x["input_ids"].to(self.device), train_Y.to(self.device))

        val_x = []
        val_Y = []
        for _, row in raw_val.iterrows():
            for s in row:
                if any(label in s for label in self.label_dict.keys()):
                    s = re.sub('(.+)(\[.*])', r'\2 \1', s)
                    s = re.sub('(\[.*])', r'\1 ', s)
                    if re.search('(\[.*])', s) is None:
                        break
                    val_x.append(" ".join(s.split()[1:]))
                    val_Y.append(self.label_dict[s.split()[0]])
        val_x = self.tokenizer.batch_encode_plus(val_x, max_length=self.input_dim, padding="max_length", truncation=True, return_tensors="pt")
        val_Y = torch.LongTensor(val_Y)
        self.val_set = TensorDataset(val_x["input_ids"].to(self.device), val_Y.to(self.device))

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, drop_last=True)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        accuracy = self.accuracy(y_pred, y)

        self.log('acc', accuracy, prog_bar=True, sync_dist=True)
        return {'loss': loss, 'acc': accuracy}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        accuracy = self.accuracy(y_pred, y)

        self.log_dict({'val_loss': loss, 'val_acc': accuracy}, prog_bar=True, sync_dist=True)
        return {'val_loss': loss, 'val_acc': accuracy}

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([output['val_loss'] for output in outputs]).mean()
        mean_acc = torch.stack([output['val_acc'] for output in outputs]).mean()

        self.log_dict({'avg_val_loss': mean_loss, 'avg_val_acc': mean_acc}, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'avg_val_loss': mean_loss, 'avg_val_acc': mean_acc}


epochs = 50
trainer = Trainer(max_epochs=epochs, gpus=torch.cuda.device_count(),
                  logger=TensorBoardLogger("../models/label_classifier/tensorboardLog/"))
for use_nll in [False, True]:
    for model_name in ["GRU", "LSTM"]:
        model = LabelClassification(epochs, use_nll=use_nll, model_name=model_name)
        trainer.fit(model)
        torch.save(model.state_dict(), "../models/label_classifier/model_state/model_state.pt")
