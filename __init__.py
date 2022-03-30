import setuptools
from transformers import GPT2LMHeadModel, BartForConditionalGeneration, PreTrainedTokenizerFast
from HaYan_NLP.model_making.persona_model.persona_classification import Persona_classifier
import numpy as np
import torch

# models
USING_TURN = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# embedding model
embedding_model = torch.nn.Embedding(51200, 768).to(device)
embedding_model.load_state_dict(torch.load("./models/embedding_model/embedding_model_state.pt"))
# chat_model
chat_model = GPT2LMHeadModel.from_pretrained("./models/gen_chat_model/trainer").to(device)
# # persona_classifier
# persona_classifier = Persona_classifier(Persona_classifier.add_model_argments(parser=None, return_nameSpace=True)).to(device)
# persona_classifier.load_state_dict(torch.load("./models/persona_classifier/best_model_state.pt", map_location=torch.device(device)))
# # persona_converter
# persona_converter = BartForConditionalGeneration.from_pretrained("./models/persona_converter/trainer").to(device)
# tokenizers
tokenizer_for_chat = PreTrainedTokenizerFast.from_pretrained("./tokenizer/GPT")
tokenizer_for_persona = PreTrainedTokenizerFast.from_pretrained("./tokenizer/koBart")
tokenizer_for_chat.encode("토크나이저 준비")
tokenizer_for_persona.encode("토크나이저 준비")

# labels
label_dict = {0: '[HAPPY]', 1: '[PANIC]', 2: '[ANGRY]', 3: '[UNSTABLE]', 4: '[HURT]', 5: '[SAD]', 6: '[NEUTRAL]'}
