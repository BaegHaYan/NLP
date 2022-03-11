import setuptools
import re
import torch
import math
import pandas as pd
import argparse
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import GPT2TokenizerFast

parser = argparse.ArgumentParser()
parser.add_argument("-e", type=int, default=50, dest="epochs", help="num of epochs")
parser.add_argument("-b", type=int, default=32, dest="batch_size", help="size of each batch")
parser.add_argument("-h", type=int, default=256, dest="hidden_size", help="size of hidden_state")
parser.add_argument("-l", type=int, default=12, dest="num_layers", help="num of transfomer model encoder layers")
parser.add_argument("-p", type=int, default=5, dest="patience", help="number of check with no improved")
parser.add_argument("-lr", type=float, default=0.01, dest="learning_rate", help="learning rate")
parser.add_argument("-dr", type=float, default=0.1, dest="dropout_rate", help="dropout rate")
parser.add_argument("-wr", type=float, default=0.05, dest="warmup_ratio", help="warmup rate")
parser.add_argument("--embedding-size", type=int, default=512, dest="embedding_size", help="size of embedding vector")

class PositionalEncoding(torch.nn.Module):
    def __init__(self, model_dim, input_dim, dropout_rate):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(dropout_rate)
        pos_encoding = torch.zeros(input_dim, model_dim)
        position_list = torch.arange(input_dim, dtype=torch.float).view(-1, 1)  # unsqueez(1)
        division_term = torch.exp(torch.arange(0, model_dim, 2).float()*(-math.log(10000)/model_dim))
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(position_list * division_term)
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(position_list * division_term)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.dropout(x + self.pos_encoding[:x.size(0)])

class LabelClassification(LightningModule):
    def __init__(self, hparams, use_nll: bool = False):
        super(LabelClassification, self).__init__()
        self.RANDOM_SEED = 7777
        pl.seed_everything(self.RANDOM_SEED)

        self.epochs = hparams.epochs
        self.batch_size = hparams.batch_size
        self.embedding_size = hparams.embedding_size
        self.hidden_size = hparams.hidden_size
        self.num_layers = hparams.num_layers
        self.patience = hparams.patience
        self.learning_rate = hparams.learning_rate
        self.dropout_rate = hparams.dropout_rate
        self.warmup_ratio = hparams.warmup_ratio

        self.use_nll = use_nll
        self.num_labels = 7
        self.input_dim = 55

        self.label_dict = {'[HAPPY]': 0, '[PANIC]': 1, '[ANGRY]': 2, '[UNSTABLE]': 3, '[HURT]': 4, '[SAD]': 5, '[NEUTRAL]': 6}
        self.train_set = None  # 3366
        self.val_set = None  # 436
        self.tokenizer = GPT2TokenizerFast.from_pretrained("../tokenizer")
        self.pad_token_id = self.tokenizer.pad_token_id

        self.embedding_layer = torch.nn.Sequential(
            torch.nn.Embedding(self.tokenizer.vocab_size, self.embedding_size, self.pad_token_id),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.ELU(),
            torch.nn.LayerNorm(self.embedding_size, eps=1e-5, elementwise_affine=True)
        )
        transformerEncoder_layer = torch.nn.TransformerEncoderLayer(self.embedding_size, 8, dropout=self.dropout_rate, batch_first=True)
        self.transformerEncoder = torch.nn.Sequential(
            PositionalEncoding(self.embedding_size, self.input_dim, self.dropout_rate),
            torch.nn.TransformerEncoder(transformerEncoder_layer, self.num_layers, norm=torch.nn.LayerNorm(self.embedding_size, eps=1e-5, elementwise_affine=True))
        )
        self.model_output_layer = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_size, self.hidden_size),
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
        lr_scheduler = {'scheduler': scheduler, 'name': 'cosine_schedule_with_warmup', 'monitor': 'loss', 'interval': 'step', 'frequency': 1}
        return [optim], [lr_scheduler]

    def configure_callbacks(self):
        model_checkpoint = ModelCheckpoint(dirpath=f"../models/label_classifier/{'nll_' if self.use_nll else ''}model_ckp/",
                                           filename='{epoch:02d}_{loss:.2f}', verbose=True, save_last=True, monitor='val_loss', mode='min')
        early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=self.patience)
        lr_monitor = LearningRateMonitor(logging_interval="step")
        return [model_checkpoint, early_stopping, lr_monitor]

    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.transformerEncoder(x)
        x = self.model_output_layer(x)
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

        self.log('loss', loss)
        self.log('acc', accuracy, prog_bar=True)
        return {'loss': loss, 'acc': accuracy}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        accuracy = self.accuracy(y_pred, y)

        self.log_dict({'val_loss': loss, 'val_acc': accuracy}, prog_bar=True)
        return {'val_loss': loss, 'val_acc': accuracy}

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([output['val_loss'] for output in outputs]).mean()
        mean_acc = torch.stack([output['val_acc'] for output in outputs]).mean()

        self.log_dict({'avg_val_loss': mean_loss, 'avg_val_acc': mean_acc}, on_epoch=True, prog_bar=True, sync_dist=True)
        return {'avg_val_loss': mean_loss, 'avg_val_acc': mean_acc}


args = parser.parse_args()
for use_nll in [False, True]:
    trainer = Trainer(max_epochs=args.epochs, gpus=torch.cuda.device_count(),
                      logger=TensorBoardLogger("../models/label_classifier/tensorboardLog/"))
    model = LabelClassification(args, use_nll=use_nll)
    trainer.fit(model)
    torch.save(model.state_dict(), f"../models/label_classifier/model_state/{'nll_' if use_nll else ''}model_state.pt")
