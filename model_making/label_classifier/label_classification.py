import setuptools
import re
import torch
import math
import pandas as pd
import argparse
import pytorch_lightning as pl
from torch.functional import F
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from torch.optim.lr_scheduler import ExponentialLR
from transformers import GPT2TokenizerFast

parser = argparse.ArgumentParser()

class InputMonitor(pl.Callback):
    def __init__(self):
        super(InputMonitor, self).__init__()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx: int, unused=0) -> None:
        if (batch_idx + 1) % trainer.log_every_n_steps == 0:
            x, y = batch
            logger = trainer.logger
            logger.experiment.add_histogram("input", x, global_step=trainer.global_step)
            logger.experiment.add_histogram("target", y, global_step=trainer.global_step)

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

class LabelClassifier(LightningModule):
    def __init__(self, hparams):
        super(LabelClassifier, self).__init__()
        self.RANDOM_SEED = 7777
        pl.seed_everything(self.RANDOM_SEED)

        self.epochs = hparams.epochs
        self.batch_size = hparams.batch_size
        self.embedding_size = hparams.embedding_size
        self.hidden_size = hparams.hidden_size
        self.num_layers = hparams.num_layers
        self.patience = hparams.patience
        self.gamma = hparams.gamma
        self.learning_rate = hparams.learning_rate
        self.dropout_rate = hparams.dropout_rate
        self.is_train = hparams.train
        self._frozen = True

        if self.is_train:
            self.unfreeze_model()
        else:
            self.eval()
            self.freeze_model()

        self.num_labels = 7
        self.input_dim = 55

        self.label_dict = {'[HAPPY]': 0, '[PANIC]': 1, '[ANGRY]': 2, '[UNSTABLE]': 3, '[HURT]': 4, '[SAD]': 5, '[NEUTRAL]': 6}
        self.train_set = None  # 3366
        self.val_set = None  # 436
        self.tokenizer = GPT2TokenizerFast.from_pretrained("../../tokenizer/GPT")
        self.pad_token_id = self.tokenizer.pad_token_id

        self.embedding_layer = torch.nn.Sequential(
            torch.nn.Embedding(self.tokenizer.vocab_size, self.embedding_size, self.pad_token_id),
            torch.nn.LayerNorm(self.embedding_size, eps=1e-5)
        )
        transformerEncoder_layer = torch.nn.TransformerEncoderLayer(self.embedding_size, 8, dropout=self.dropout_rate, activation=F.relu, batch_first=True)
        self.transformerEncoder = torch.nn.Sequential(
            PositionalEncoding(self.embedding_size, self.input_dim, self.dropout_rate),
            torch.nn.TransformerEncoder(transformerEncoder_layer, self.num_layers, norm=torch.nn.LayerNorm(self.embedding_size, eps=1e-5, elementwise_affine=True)),
        )
        self.model_output_layer = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_size, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(self.hidden_size, eps=1e-5, elementwise_affine=True),
            torch.nn.Dropout(self.dropout_rate)
        )
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size*2),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden_size*2, self.hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden_size, self.num_labels)
        )

    def configure_optimizers(self):
        adam = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.1)
        rms_prop = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate, weight_decay=0.1)

        lr_scheduler_adam = ExponentialLR(adam, gamma=self.gamma)
        lr_scheduler_rms = ExponentialLR(rms_prop, gamma=self.gamma)
        return [adam, rms_prop], [lr_scheduler_adam, lr_scheduler_rms]

    def configure_callbacks(self):
        model_checkpoint = ModelCheckpoint(dirpath=f"../models/label_classifier/model_ckp/",
                                           filename='{epoch:02d}_{loss:.2f}', verbose=True, save_last=True, monitor='val_loss', mode='min')
        early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=self.patience)
        lr_monitor = LearningRateMonitor(logging_interval="step")
        input_monitor = InputMonitor()
        return [model_checkpoint, early_stopping, lr_monitor, input_monitor]

    def unfreeze_model(self):
        if self._frozen:
            for param in self.parameters():
                param.requires_grad = True
            self._frozen = False

    def freeze_model(self):
        for param in self.parameters():
            param.requires_grad = False
        self._frozen = True

    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.transformerEncoder(x)
        x = self.model_output_layer(x)
        output = self.output_layer(x)
        return F.softmax(torch.sum(output, 1), 1)

    def loss(self, output, labels):
        return torch.nn.CrossEntropyLoss(ignore_index=self.pad_token_id)(output, labels)

    def accuracy(self, output, labels):
        output = torch.argmax(output, dim=1)
        return torch.sum(output == labels) / output.__len__() * 100  # %(Precentage)

    def prepare_data(self):
        raw_train = pd.read_csv("../../data/label_classification_dataset/train/train_data1.txt", sep="\t", encoding="949").drop(["R1", "R2", "R3"], axis=1)
        raw_val = pd.read_csv("../../data/label_classification_dataset/val/val_data1.txt", sep="\t", encoding="949").drop(["R1", "R2", "R3"], axis=1)

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

        self.log_dict({'avg_val_loss': mean_loss, 'avg_val_acc': mean_acc}, on_epoch=True, prog_bar=True)
        return {'avg_val_loss': mean_loss, 'avg_val_acc': mean_acc}

    @staticmethod
    def set_hparam(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument("-e", type=int, default=50, dest="epochs", help="num of epochs")
        parser.add_argument("-b", type=int, default=32, dest="batch_size", help="size of each batch")
        parser.add_argument("-hd", type=int, default=256, dest="hidden_size", help="size of hidden_state")
        parser.add_argument("-l", type=int, default=12, dest="num_layers", help="num of transfomer model encoder layers")
        parser.add_argument("-p", type=int, default=5, dest="patience", help="number of check with no improved")
        parser.add_argument("-lr", type=float, default=0.1, dest="learning_rate", help="learning rate")
        parser.add_argument("-dr", type=float, default=0.1, dest="dropout_rate", help="dropout rate")
        parser.add_argument("-gamma", type=float, default=0.9, dest="gamma", help="decay rate of learning_rate on each epoch")
        parser.add_argument("--embedding-size", type=int, default=512, dest="embedding_size", help="size of embedding vector")
        parser.add_argument("-training", type=bool, default=False, dest="train", help="condiiton about this model for training")
        return parser

    def predict(self, x):
        return self(x)


parser = LabelClassifier.set_hparam(parser)
args = parser.parse_args()
trainer = Trainer(max_epochs=args.epochs, gpus=torch.cuda.device_count(), logger=TensorBoardLogger(
    "../../models/label_classifier/tensorboardLog/"))
model = LabelClassifier(args)
trainer.fit(model)
torch.save(model.state_dict(), "../models/label_classifier/model_state/model_state.pt")
