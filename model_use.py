import __init__
import numpy as np
from typing import List

def use_chat_model(text: str) -> str:
    text = __init__.p.encoding(text)
    output = __init__.chat_model.generate(text,
                                          max_length=1000,
                                          num_beams=5,
                                          top_k=20,
                                          no_repeat_ngram_size=4,
                                          length_penalty=0.65,
                                          repetition_penalty=2.)
    return __init__.p.decoding(output[0][text.shape[-1]:])

def compress_sent(sent: str) -> List[float]:
    encoded_sent = __init__.p.compress_tokenizer.encode(sent, return_tensors="tf")
    return __init__.compress_model(encoded_sent)["pooler_output"].numpy().tolist()

def cos_sim(v1: List[float], v2: List[float]) -> np.float:
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
