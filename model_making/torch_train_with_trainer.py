from transformers import GPT2LMHeadModel, Trainer, TrainingArguments
from preprocessing import Preprocesser
import datetime
import torch
import os

if __name__ == "__main__":
    class koDialoGPT(torch.nn.Module):
        def __init__(self):
            super(koDialoGPT, self).__init__()
            self.model = GPT2LMHeadModel.from_pretrained(p.PREMODEL_NAME)

        def forward(self, x):
            input_ids, attention_mask = x
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            return logits.logits

    def compute_metric():
        pass

    p = Preprocesser()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = koDialoGPT().to(device)
    epochs = 5

    log_dir = os.path.join('./logs', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    train_args = TrainingArguments(output_dir="../model/torch_models/traniner_model", logging_dir=log_dir, num_train_epochs=epochs)

    trainer = Trainer(model=model, args=train_args, compute_metrics=None,
                      train_dataset=p.getTorchTrainData(), eval_dataset=p.getTorchValidationData())
    trainer.train()
