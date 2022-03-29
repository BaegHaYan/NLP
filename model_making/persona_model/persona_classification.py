import setuptools
import torchmetrics
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import SequentialLR, ConstantLR, ExponentialLR
from transformers import GPT2TokenizerFast
import torch
import argparse
import logging
import pandas as pd

class Persona_classification(torch.nn.Module):
    def __init__(self, hparams: argparse.Namespace):
        super(Persona_classification, self).__init__()
        self.tokenizer = GPT2TokenizerFast.from_pretrained("../../tokenizer/GPT")

        self.input_dim = 30  # consider changing model to LSTM/CNN(input has short dim)
        self.embedding_dim = hparams.embedding_dim
        self.hidden_dim = hparams.hidden_dim
        self.nhead = hparams.nhead
        self.num_layers = hparams.num_layers
        self.dim_feedforward = hparams.dim_feedforward
        self.vocab_size = 51200
        self.pad_token_id = 3

        self.embed_layer = torch.nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.pad_token_id)
        transformer_encoder = torch.nn.TransformerEncoderLayer(self.embedding_dim, self.nhead, dim_feedforward=self.dim_feedforward, batch_first=True)
        self.transformer_encoder_layer = torch.nn.TransformerEncoder(transformer_encoder, num_layers=self.num_layers, norm=torch.nn.LayerNorm(self.embedding_dim))
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, int(self.hidden_dim/2)),
            torch.nn.Tanh(),
            torch.nn.LayerNorm(int(self.hidden_dim/2)),
            torch.nn.Linear(int(self.hidden_dim/2), 2)
        )

    def forward(self, x):
        x = self.embed_layer(x)
        x = self.transformer_encoder_layer(x)
        output = self.output_layer(x)
        return output

    def predict(self, sent: str):
        input_dim = 30
        x = self.tokenizer.encode(sent, max_length=input_dim, padding="max_length", truncation=True)["input_ids"]

        output = self(x)
        output = output.view(1, -1)
        output = torch.sum(output, dim=1)
        output = torch.sigmoid(output)
        return output

    @staticmethod
    def add_model_argments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument("--embedding-dim", type=int, default=960, dest="embedding_dim", help="embedding size")
        parser.add_argument("--hidden-size", type=int, default=786, dest="hidden_dim", help="hidden size")
        parser.add_argument("--attention-head", type=int, default=12, dest="nhead", help="num of attention heads")
        parser.add_argument("--num-layers", type=int, default=12, dest="num_layers", help="num of layers")
        parser.add_argument("--dim-feedforward", type=int, default=2048, dest="dim_feedforward", help="dim of feedforward")
        return parser

def get_dataset(tokenizer: GPT2TokenizerFast, hparams: argparse.Namespace, logger: logging.Logger, device: str):
    input_dim = 30
    train_path = "../../data/persona_dataset/data1.txt"
    val_path = "../../data/persona_dataset/data1.txt"

    datasets = []
    for i, path in enumerate([train_path, val_path]):
        data = pd.read_csv(path, sep="\t", encoding="949")
        data.dropna(inplace=True)
        x = []
        Y = []

        x.extend(tokenizer.batch_encode_plus(data["non_persona"].to_list(), max_length=input_dim, padding="max_length", truncation=True)["input_ids"])
        x.extend(tokenizer.batch_encode_plus(data["persona"].to_list(), max_length=input_dim, padding="max_length", truncation=True)["input_ids"])
        Y.extend([0] * len(data["non_persona"]))
        Y.extend([1] * len(data["persona"]))
        if i == 0:
            x *= 100
            Y *= 100
        x = torch.LongTensor(x).to(device)
        Y = torch.FloatTensor(Y).to(device)
        datasets.append(TensorDataset(x, Y))
    train_dataloader = DataLoader(datasets[0], batch_size=hparams.batch_size, shuffle=True, drop_last=True)
    logger.info("making train dataloader complete")
    val_dataloader = DataLoader(datasets[1], batch_size=hparams.batch_size, shuffle=True, drop_last=True)
    logger.info("making val dataloader complete")

    return train_dataloader, val_dataloader


if __name__ == "__main__":
    RANDOM_SEED = 777
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser()
    parser = Persona_classification.add_model_argments(parser)
    parser.add_argument("-e", type=int, default=100, dest="epochs", help="epochs")
    parser.add_argument("-b", type=int, default=32, dest="batch_size", help="batch_size")
    parser.add_argument("-lr", type=float, default=0.01, dest="lr", help="learning rate")
    parser.add_argument("-factor", type=float, default=0.5, dest="factor", help="factor (multiplied number by the first few epochs)")
    parser.add_argument("-gamma", type=float, default=0.9, dest="gamma", help="gamma (decay rate for each epoch)")
    parser.add_argument("-warmup_rate", type=float, default=0.01, dest="warmup_rate", help="warmup_rate")
    hparams = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter())
    logger.addHandler(handler)

    model = Persona_classification(hparams).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained("../../tokenizer/GPT")
    accuracy = torchmetrics.Accuracy().to(device)
    loss_func = torch.nn.BCELoss().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=hparams.lr)

    constant_scheduler = ConstantLR(optim, factor=hparams.factor)
    exponential_scheduler = ExponentialLR(optim, gamma=hparams.gamma)
    lr_scheduler = SequentialLR(optim, schedulers=[constant_scheduler, exponential_scheduler], milestones=[int(hparams.epochs * hparams.warmup_rate)])

    best_metrics = 0
    train_loader, val_loader = get_dataset(tokenizer, hparams, logger, device)
    for i in range(hparams.epochs):
        for j, batch in enumerate(train_loader):
            optim.zero_grad()

            x, Y = batch
            pred = model(x)
            pred = pred.view(hparams.batch_size, -1)
            pred = torch.sum(pred, dim=1)
            pred = torch.sigmoid(pred)

            loss = loss_func(pred, Y)
            acc = accuracy(pred, Y.type(torch.int))
            if j % 10 == 0:
                logger.info(f"Epochs {i+1} : loss - %.4f, acc - %.4f | {j+1}/{len(train_loader)}(%.2f%%)"
                            % (loss.item(), acc.item()*100, (j+1)/len(train_loader) * 100))
            loss.backward()
            optim.step()

        with torch.no_grad():
            val_loss = []
            val_acc = []
            for batch in val_loader:
                x, Y = batch
                pred = model(x)
                pred = pred.view(hparams.batch_size, -1)
                pred = torch.sum(pred, dim=1)
                pred = torch.sigmoid(pred)

                val_loss.append(loss_func(pred, Y).item())
                val_acc.append(accuracy(pred, Y.type(torch.int)).item())
            val_loss = torch.mean(torch.tensor(val_loss))
            val_acc = torch.mean(torch.tensor(val_acc)) * 100
            logger.info(f"Epochs {i+1} : val_loss - %.4f, val_acc - %.4f)" % (val_loss.item(), val_acc.item()))

            if val_acc > best_metrics:
                logger.info(f"val_loss has achieved the best : {best_metrics} to {val_acc}")
                best_metrics = val_acc

                torch.save(model.state_dict(), "../../models/persona_classifier/best_model_state.pt")
                logger.info("the model state was saved.")
            else:
                logger.info(f"the model state was NOT achieved the best. | the best - {best_metrics}")

        lr_scheduler.step()
