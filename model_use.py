from typing import List
import numpy as np
import __init__
import torch

using_turn = 2
max_length = 1000
eos_token = __init__.p.tokenizer.eos_token
def use_chat_model(user_text: str, past_convs: List[str]) -> str:
    user_text = __init__.p.encoding(user_text+eos_token)
    for i in range(len(past_convs)-1, len(past_convs)-(1+using_turn*2), -1):
        if i < 0:
            break
        encoded_vector = __init__.p.encoding(past_convs[i]+eos_token)
        if user_text.shape[-1] + encoded_vector.shape[-1] < max_length:
            user_text = torch.cat([encoded_vector, user_text], dim=-1)
        else:
            break

    user_text = user_text.to(__init__.device)
    output = __init__.chat_model.generate(user_text,
                                          max_length=1000,
                                          num_beams=5,
                                          top_k=20,
                                          no_repeat_ngram_size=4,
                                          length_penalty=0.65,
                                          repetition_penalty=2.)
    numpy_output = output.numpy()
    numpy_output = np.where(numpy_output == 25856, 389, numpy_output)
    numpy_output = np.where(numpy_output == 7098, 8142, numpy_output)
    numpy_output = np.where(numpy_output == 25830, 5, numpy_output)
    output = torch.from_numpy(numpy_output)

    return __init__.p.decoding(output[0][user_text.shape[-1]:])
