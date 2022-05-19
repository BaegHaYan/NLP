from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import torch
import time

class hps:
    RANDOM_SEED = 7777
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    PREMODEL_NAME = "byeongal/Ko-DialoGPT"
    src_max_len = None
    trg_max_len = None
    num_workers = torch.cuda.device_count() if device == "cuda" else 2
    pad_token_id = 3

    epochs = 4
    batch_size = 16
    lr = 3e-5
    weight_decay = 1e-6


def getDataset(data_type: str, tokenizer) -> torch.utils.data.DataLoader:
    torch.manual_seed(hps.RANDOM_SEED)
    torch.cuda.manual_seed(hps.RANDOM_SEED)

    data_path = "../../data/train.txt" if data_type == "train" else data_path = "../../data/val.txt"

    data = pd.read_csv(data_path, sep="\t", encoding="utf-8", index_col=0)
    q = tokenizer.batch_encode_plus(data["Q"].to_list(), max_length=hps.src_max_len, padding="max_length",
                                    truncation=True, return_tensors="pt").to(hps.device)
    a = tokenizer.batch_encode_plus(data["A"].to_list(), max_length=hps.trg_max_len, padding="max_length",
                                    truncation=True, return_tensors="pt")["input_ids"].to(hps.device)

    return DataLoader(TensorDataset((q, a)),
                      batch_size=hps.batch_size, shuffle=(data_type == "train"),
                      num_workers=hps.num_workers, drop_last=True)


if __name__ == "__main__":
    GPTtokenizer = GPT2TokenizerFast.from_pretrained("../../tokenizer/GPT")
    model = GPT2LMHeadModel.from_pretrained(hps.PREMODEL_NAME).to(hps.device)
    model.transformer.requires_grad_(False)
    model.lm_head.requires_grad_(True)

    train_loader = getDataset("train", GPTtokenizer)
    val_loader = getDataset("val", GPTtokenizer)

    optimizer = torch.optim.Adam(model.lm_head.parameters(), lr=hps.lr, weight_decay=hps.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=hps.pad_token_id)

    min_loss = 99
    for i in range(hps.epochs):
        for j, batch in enumerate(train_loader):
            start = time.perf_counter()
            x, y = batch
            y_pred = model(x)

            # loss
            loss = criterion(y_pred, y)

            # zero grad
            model.zero_grad()

            # backward, grad_norm, and update
            loss.backward()
            optimizer.step()

            dur = time.perf_counter() - start
            # info
            print('Epoch {}, {}/{}({:.2f}%): loss: {} {:.1f}s/it'.format(i, j, len(train_loader), j/len(train_loader)*100, loss.item(), dur))
            if loss.item() < min_loss:
                print(f"saved model: metrics improved {min_loss} to {loss.item()}.")
                min_loss = loss.item()
                torch.save(model, "../../models/gen_chat_model/pytorch_model.bin")


