from transformers import GPT2LMHeadModel
from preprocessing import Preprocesser
import torch

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.device(device)
    p = Preprocesser()
    print(f'Using {device} device')


    def Perplexity(predict: torch.Tensor) -> float:
        # torch.Size([16, 201, 51200]), torch.Size([16, 201])
        # ^size√(1/sum(word's probability))
        size = predict.size()[1]
        perplexed = 0
        for sent, sent2 in zip(predict, torch.argmax(predict, dim=-1)):
            for tensor, idx in zip(sent, sent2):
                perplexed += tensor[idx].item()
        return (1 / perplexed) ** (1 / size)


    class koDialoGPT(torch.nn.Module):
        def __init__(self):
            super(koDialoGPT, self).__init__()
            self.model = GPT2LMHeadModel.from_pretrained(p.PREMODEL_NAME)

        def forward(self, x):
            input_ids, attention_mask = x
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            return logits.logits


    def train_loop(dataloader, t_model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        correct = 0
        loss_ = None
        for batch, (input_ids, attention_mask, y) in enumerate(dataloader):
            y = y.to(device)
            pred = t_model((input_ids.to(device), attention_mask.to(device)))
            pred = pred.to(device)
            loss_ = loss_fn(pred.transpose(2, 1).float(), y.long())
            correct = Perplexity(pred.to('cpu'))

            optimizer.zero_grad()
            loss_.backward(gradient=loss_)
            optimizer.step()

            if batch % 10 == 0:
                loss_ = loss_.sum() / loss_.size()[-1]
                print(f"loss: {loss_:>7f}  PPL: {correct:>0.3f} [{batch:>5d}/{len(dataloader):>5d}]")
        return f"{loss_:>7f}", f"{correct :>0.3f})"


    def val_loop(dataloader, t_model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for input_ids, attention_mask, y in dataloader:
                y = y.to(device)
                pred = t_model((input_ids.to(device), attention_mask.to(device)))
                pred = pred.to(device)
                temp_loss = loss_fn(pred.transpose(2, 1).float(), y.long())
                test_loss += temp_loss.sum() / temp_loss.size()[-1]
                correct += Perplexity(pred.to('cpu'))

        test_loss /= num_batches
        correct /= size
        print(f"Validation : loss: {test_loss:>7f}  PPL: {correct:>0.3f}")
        return f"{test_loss:>7f}", f"{correct:>0.3f})"


    lr = 5e-5
    epochs = 5
    lowest_loss = 9999
    lowest_val_loss = 9999
    loss_func = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=p.tokenizer.pad_token_id)  # pad_token == 3
    history = ""
    for optim_order in [torch.optim.Adam, torch.optim.RMSprop]:
        model = koDialoGPT().to(device)
        optim = optim_order(model.parameters(), lr=lr)

        print(f"{str(optim).split()[0]} Start.")
        history += str(optim).split()[0] + "\n"
        history_list = {"loss": list(), "ppl": list(), "val_loss": list(), "val_ppl": list()}

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[2, 4], gamma=0.5)
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            loss, ppl = train_loop(model.p.getTorchTrainData(), model, loss_func, optim)
            val_loss, val_ppl = val_loop(model.p.getTorchValidationData(), model, loss_func)
            scheduler.step()

            if float(val_loss) < lowest_val_loss:
                torch.save(model, "../model/torch_models/val_loss/best_model.pt")
                torch.save(model, "../model/torch_models/val_loss/pytorch_model.bin")
                lowest_val_loss = float(val_loss)
                print("Model Saved(val_loss)")
            if float(loss) < lowest_loss:
                torch.save(model, "../model/torch_models/loss/best_model.pt")
                torch.save(model, "../model/torch_models/loss/pytorch_model.bin")
                lowest_loss = float(loss)
                print("Model Saved(loss)")

            history_list["loss"].append(loss)
            history_list["ppl"].append(ppl)
            history_list["val_loss"].append(val_loss)
            history_list["val_ppl"].append(val_ppl)

        for k, v in history_list.items():
            history += k + " : " + str(v) + "\n"
        history += "\n"
        print("Done!\n")

    open("../model/history.txt", "w+", encoding="utf-8").write(history)
