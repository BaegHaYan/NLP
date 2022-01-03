from HaYan_NLP.model_making.preprocessing import Preprocesser
from transformers import GPT2LMHeadModel
import torch

# preprocesser
p = Preprocesser()
# model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# chat_model = GPT2LMHeadModel.from_pretrained("byeongal/Ko-DialoGPT").to(device)
chat_model = GPT2LMHeadModel.from_pretrained("./model/hf_form").to(device)
# compress_model = TFBertModel.from_pretrained(p.COMPRESS_MODEL_NAME, from_pt=True, use_cache=True, cache_dir="./model/compress")
