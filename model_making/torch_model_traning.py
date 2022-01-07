import argparse
import logging
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

TOKENIZER = PreTrainedTokenizerFast.from_pretrained("../tokenizer", bos_token='<s>', eos_token='</s>')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--train',
                    action='store_true',
                    default=True,
                    help='for training')
parser.add_argument('--chat',
                    action='store_true',
                    default=False,
                    help='response generation on given user input')
parser.add_argument('--model_params',
                    type=str,
                    default='../torch_models/model_chp',
                    help='model binary for starting chat')

def add_model_specific_args(parent_parser):
    # add model specific args
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--max-len',
                        type=int,
                        default=201,
                        help='max sentence length on input (default: 201)')
    parser.add_argument('--batch-size',
                        type=int,
                        default=16,
                        help='batch size for training (default: 16)')
    parser.add_argument('--lr',
                        type=float,
                        default=3e-5,
                        help='The initial learning rate')
    parser.add_argument('--warmup_ratio',
                        type=float,
                        default=0.1,
                        help='warmup ratio')
    return parser


class dataset(Dataset):
    def __init__(self, chats: pd.DataFrame):
        self._data = chats
        self.max_len = 201
        self.tokenizer = TOKENIZER

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        turn = self._data.iloc[idx]
        q = turn['dialogue']
        a = turn['response']
        q_toked = self.tokenizer.tokenize(q)
        a_toked = self.tokenizer.tokenize(a)
        q_len = len(q_toked)
        a_len = len(a_toked)
        # token ids
        token_ids = self.tokenizer.convert_tokens_to_ids(q_toked + a_toked)
        while len(token_ids) < self.max_len:
            token_ids += [self.tokenizer.pad_token_id]
        # mask
        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)
        # labels
        labels = [self.tokenizer.mask_token] * q_len + a_toked
        labels_ids = self.tokenizer.convert_tokens_to_ids(labels)
        while len(labels_ids) < self.max_len:
            labels_ids += [self.tokenizer.pad_token_id]

        return token_ids, np.array(mask), labels_ids


class KoGPT2Chat(LightningModule):
    def __init__(self, hparams, **kwargs):
        super(KoGPT2Chat, self).__init__()
        self.hparams_ = hparams
        self.neg = -1e18
        self.train_set = None
        self.koDialoGPT = GPT2LMHeadModel.from_pretrained("byeongal/Ko-DialoGPT")
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs):
        output = self.koDialoGPT(inputs, return_dict=True)
        return output.logits

    def training_step(self, batch, batch_idx):
        token_ids, mask, label = batch
        out = self(token_ids)
        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
        mask_out = torch.where(mask_3d == 1, out, self.neg * torch.ones_like(out))
        loss = self.loss_function(mask_out.transpose(2, 1), label)
        loss_avg = loss.sum() / mask.sum()
        self.log('train_loss', loss_avg)
        return loss_avg

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams_.lr, correct_bias=False)

        return [optimizer]

    def _collate_fn(self, batch):
        data = np.array([item[0] for item in batch])
        mask = np.array([item[1] for item in batch])
        label = np.array([item[2] for item in batch])

        return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)

    def train_dataloader(self):
        data = pd.read_csv('../data/train.txt', sep="\t", names=["dialogue", "response"], header=0)
        self.train_set = dataset(data)
        train_dataloader = DataLoader(
            self.train_set, batch_size=self.hparams_.batch_size, num_workers=2,
            shuffle=True, collate_fn=self._collate_fn)
        return train_dataloader

    def chat(self, emotion='[NATURAL]'):
        tokenizer = TOKENIZER
        print("quit 입력시 종료")
        with torch.no_grad():
            while True:
                q = input('user > ').strip()
                if q == 'quit':
                    break
                a = ''
                while True:
                    input_ids = torch.LongTensor(tokenizer.encode(emotion + q + "</s>" + a)).unsqueeze(dim=0)
                    pred = self(input_ids)
                    gen = tokenizer.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]
                    if gen == "</s>":
                        break
                    a += gen.replace('▁', ' ')
                print(f"bot >> {a.strip()}")


parser = add_model_specific_args(parser)
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()
logging.info(args)

if __name__ == "__main__":
    if args.train:
        checkpoint_callback = ModelCheckpoint(
            dirpath='../torch_models/model_chp',
            filename='{epoch:02d}-{train_loss:.2f}',
            verbose=True,
            save_last=True,
            monitor='train_loss',
            mode='min'
        )
        # usage = python file.py --gpus=1 --max_epochs=10
        model = KoGPT2Chat(args)
        model.train()
        trainer = Trainer.from_argparse_args(
            args,
            checkpoint_callback=checkpoint_callback, gradient_clip_val=1.0)
        trainer.fit(model)
        logging.info('best model path {}'.format(checkpoint_callback.best_model_path))
        torch.save(model, "../model/torch_models/pytorch_model.bin")
    if args.chat:
        model = KoGPT2Chat.load_from_checkpoint(args.model_params)
        model.chat()
