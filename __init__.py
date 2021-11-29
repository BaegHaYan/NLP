from transformers import TFAutoModelForCausalLM
from preprocessing import Preprocesser

# preprocesser
p = Preprocesser()
# model
# model = TFAutoModelForCausalLM.from_pretrained(p.PREMODEL_NAME)
model = None
