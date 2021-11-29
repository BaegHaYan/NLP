import __init__

def use_model(text: str):
    text = __init__.p.encoding(text)
    output = __init__.model.generate(text,
                                     max_length=1000,
                                     num_beams=5,
                                     top_k=20,
                                     no_repeat_ngram_size=4,
                                     length_penalty=0.65,
                                     repetition_penalty=2.)
    return __init__.p.decoding(output[0][text.shape[-1]:])
