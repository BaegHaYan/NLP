from HaYan_NLP.preprocessing import Preprocesser
from transformers import GPT2LMHeadModel, BertModel
import torch

# preprocesser
p = Preprocesser()
# model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
chat_model = GPT2LMHeadModel.from_pretrained("./model/hf_form").to(device)
compress_model = BertModel.from_pretrained(p.COMPRESS_MODEL_NAME, use_cache=True, cache_dir="./model/compress")
