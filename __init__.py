from transformers import TFAutoModelForCausalLM, TFBertModel
from preprocessing import Preprocesser

# preprocesser
p = Preprocesser()
# model
chat_model = None
compress_model = TFBertModel.from_pretrained(p.COMPRESS_MODEL_NAME, from_pt=True, use_cache=True,
                                             cache_dir="./model/compress")
