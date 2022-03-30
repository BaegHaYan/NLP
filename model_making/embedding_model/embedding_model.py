from transformers import GPT2LMHeadModel
import torch

model = GPT2LMHeadModel.from_pretrained("../../models/gen_chat_model/trainer")
embedding_model = torch.nn.Embedding(51200, 768)
embedding_model.load_state_dict({"weight": model.state_dict()["transformer.wte.weight"]})
torch.save(embedding_model.state_dict(), "../../models/embedding_model/embedding_model_state.pt")
