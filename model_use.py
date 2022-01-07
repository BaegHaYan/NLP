from typing import List, Dict
import numpy as np
import __init__
import torch

USING_TURN = 2
EOS = __init__.p.tokenizer.eos_token
def chat_with_model(user_text: str, past_convs: List[str]) -> str:
    user_text = __init__.p.encoding(user_text+EOS)
    for i in range(len(past_convs)-1, len(past_convs)-(1+USING_TURN*2), -1):
        if i < 0:
            break
        encoded_vector = __init__.p.encoding(past_convs[i]+EOS)
        user_text = torch.cat([encoded_vector, user_text], dim=-1)

    user_text = user_text.to(__init__.device)
    output = __init__.chat_model.generate(user_text,
                                          max_length=1000,
                                          num_beams=5,
                                          top_k=20,
                                          no_repeat_ngram_size=4,
                                          length_penalty=0.65,
                                          repetition_penalty=2.)

    return __init__.p.decoding(output[0][user_text.shape[-1]:])

def searching_QnA(user_text: str, DB_QnA: Dict[str, str]) -> str:
    max_sim = 0.9
    searched_answer = None

    def compress_sent(sent: str) -> List[float]:
        encoded_sent = __init__.p.compress_tokenizer.encode(sent, return_tensors="tf")
        return __init__.compress_model(encoded_sent)["pooler_output"].numpy().tolist()

    def cos_sim(v1: List[float], v2: List[float]) -> np.float:
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    user_text = compress_sent(user_text)
    for question, answer in DB_QnA.items():
        temp_sim = cos_sim(user_text, compress_sent(question))
        if temp_sim > max_sim:
            max_sim = temp_sim
            searched_answer = answer

    return searched_answer
