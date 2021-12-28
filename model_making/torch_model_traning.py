from transformers import GPT2LMHeadModel
from preprocessing import Preprocesser
from sklearn.metrics import accuracy_score
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.device(device)
print(f'Using {device} device')

class koDialoGPT(torch.nn.Module):
    def __init__(self):
        super(koDialoGPT, self).__init__()
        self.p = Preprocesser()
        self.model = GPT2LMHeadModel.from_pretrained(self.p.PREMODEL_NAME)

    def forward(self, x):
        input_ids, attention_mask = x
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return logits


def train_loop(dataloader, t_model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    correct = 0
    loss_ = None
    for batch, (input_ids, attention_mask, y) in enumerate(dataloader):
        pred = t_model((input_ids, attention_mask))
        loss_ = loss_fn(pred, y)
        correct += accuracy_score(y, torch.max(pred, dim=-1))

        optimizer.zero_grad()
        loss_.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss_, current = loss_.item(), batch * len(input_ids)
            print(f"loss: {loss_:>7f}  Accuracy: {correct/size * 100:>0.1f} [{current:>5d}/{size:>5d}]")
    return f"{loss_:>7f}", f"{correct / size * 100:>0.1f})"

def val_loop(dataloader, t_model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = t_model(X)
            test_loss += loss_fn(pred, y).item()
            correct += accuracy_score(y, torch.max(pred, dim=-1))

    test_loss /= num_batches
    correct /= size
    print(f"Validation : Accuracy: {(100*correct):>0.1f}%, loss: {test_loss:>8f} \n")
    return f"{test_loss:>8f}", f"{(100*correct):>0.1f})"


lr = 5e-5
epochs = 5
highscore = 0
model = koDialoGPT().to(device)
loss_func = torch.nn.CrossEntropyLoss()

history = ""
for optim in [torch.optim.Adam(model.parameters(), lr=lr), torch.optim.RMSprop(model.parameters(), lr=lr)]:
    print(f"{str(optim).split()[0]} Start.")
    history += str(optim).split()[0] + "\n"
    history_list = {"loss": list(), "acc": list(), "val_loss": list(), "val_acc": list()}

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[2, 4], gamma=0.5)
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        loss, acc = train_loop(model.p.getTorchTrainData(), model, loss_func, optim)
        val_loss, val_acc = val_loop(model.p.getTorchValidationData(), model, loss_func)
        scheduler.step()

        if float(val_acc) > highscore:
            torch.save(model, "../model/torch_models/best_model.pt")
            torch.save(model, "../model/torch_models/pytorch_model.bin")

        history_list["loss"].append(loss)
        history_list["acc"].append(acc)
        history_list["val_loss"].append(val_loss)
        history_list["val_acc"].append(val_acc)

    for k, v in history_list.items():
        history += k + " : " + str(v) + "\n"
    history += "\n"
    print("Done!\n")

open("../model/history.txt", "w+", encoding="utf-8").write(history)
