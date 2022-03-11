import setuptools
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

    def forward(self, x: torch.tensor, encoder_last_state: torch.tensor = None) -> torch.tensor:
        return self.dropout(x + self.pos_encoding[:x.size(0)]), encoder_last_state

class ChatModel(LightningModule):
    def __init__(self, hparams):
        super(ChatModel, self).__init__()
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

        self.tokenizer = GPT2TokenizerFast.from_pretrained("../../tokenizer")
        self.pad_token_id = self.tokenizer.pad_token_id
        self.src_dim = None
        self.target_dim = None
        self.train_set = None
        self.val_set = None
        self.test_set = None

        self.embedding_layer = torch.nn.Embedding(self.tokenizer.vocab_size, self.embedding_size, self.pad_token_id)
        transformerEncoder_layer = torch.nn.TransformerEncoderLayer(self.embedding_size, 8, dropout=self.dropout_rate, batch_first=True)
        transformerDecoder_layer = torch.nn.TransformerDecoderLayer(self.embedding_size, 8, dropout=self.dropout_rate, batch_first=True)
        self.transformerEncoder = torch.nn.Sequential(
            PositionalEncoding(self.embedding_size, self.scr_dim, self.dropout_rate),
            torch.nn.TransformerEncoder(transformerEncoder_layer, self.num_layers, norm=torch.nn.LayerNorm(self.embedding_size, eps=1e-5, elementwise_affine=True))
        )
        self.transformerDecoder = torch.nn.Sequential(
            PositionalEncoding(self.embedding_size, self.target_dim, self.dropout_rate),  # TODO check return is not problem
            torch.nn.TransformerDecoder(transformerDecoder_layer, self.num_layers, norm=torch.nn.LayerNorm(self.embedding_size, eps=1e-5, elementwise_affine=True))
        )
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_size, self.hidden_size),  # TODO check the shape of decoder output
            torch.nn.ELU(),
            torch.nn.LayerNorm(self.hidden_size, eps=1e-5, elementwise_affine=True),
            torch.nn.Dropout(self.dropout_rate),
            torch.nn.LogSoftmax(dim=1)  # TODO check dim
        )

    def configure_optimizers(self):
        optim = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.1)

        num_train_steps = len(self.train_dataloader()) * self.epochs
        num_warmup_steps = int(num_train_steps * self.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 'name': 'cosine_schedule_with_warmup', 'monitor': 'loss', 'interval': 'step', 'frequency': 1}
        return [optim], [lr_scheduler]

    def configure_callbacks(self):
        model_checkpoint = ModelCheckpoint(dirpath=f"../../models/chat_model/model_ckp/",
                                           filename='{epoch:02d}_{loss:.2f}', verbose=True, save_last=True, monitor='val_loss', mode='min')
        early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=self.patience)
        lr_monitor = LearningRateMonitor(logging_interval="step")
        return [model_checkpoint, early_stopping, lr_monitor]

    def forward(self, input_e, input_d):  # TODO return values
        input_e = self.embedding_layer(input_e)
        encoder_last_state = self.transformerEncoder(input_e)
        input_d = self.embedding_layer(input_d)
        x = self.transformerDecoder(input_d, encoder_last_state)
        output = self.output_layer(x)
        return output

    def NLLloss(self, output, labels):
        return torch.nn.NLLLoss()(output, labels)

    def prepare_data(self):
        pass  # TODO

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, drop_last=True)

    def training_step(self, batch, batch_idx):
        input_e, input_d, y = batch
        y_pred = self(input_e, input_d)
        loss = self.NLLloss(y_pred, y)

        self.log('loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        input_e, input_d, y = batch
        y_pred = self(input_e, input_d)
        loss = self.NLLloss(y_pred, y)

        self.log('val_loss', loss)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([output['val_loss'] for output in outputs]).mean()

        self.log('avg_val_loss', mean_loss, on_epoch=True, prog_bar=True)
        return {'avg_val_loss': mean_loss}


args = parser.parse_args()
trainer = Trainer(max_epochs=args.epochs, gpus=torch.cuda.device_count(),
                  logger=TensorBoardLogger("../../models/chat_model/tensorboardLog/"))
model = ChatModel(args)
trainer.fit(model)
torch.save(model.state_dict(), "../../models/chat_model/model_state/model_state.pt")
