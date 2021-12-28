from transformers import TFGPT2LMHeadModel
from preprocessing import Preprocesser
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--once", type=bool, default=False, metavar="Bool", dest="is_once")

if __name__ == "__main__":
    p = Preprocesser()
    model = TFGPT2LMHeadModel.from_pretrained("../model/hf_form")

    if parser.parse_args().is_once:
        parser.add_argument('-text', '--text', type=str, default=input("User >>"), metavar='str', dest="text")
        text = parser.parse_args().text

        output = model.generate(p.encoding(text),
                                max_length=1000,
                                num_beams=5,
                                top_k=20,
                                no_repeat_ngram_size=4,
                                length_penalty=0.65,
                                repetition_penalty=2.
                                )
        print(p.decoding(output))
    else:
        print("quit이라고 입력하면 종료.")
        print("'[감정] 대사' 형태로 입력해야 함.")
        turn_text = input("User >> ")
        input_text = p.tokenizer.bos_token
        while turn_text != "quit":
            input_text += turn_text + p.tokenizer.eos_token
            if len(input_text.split(p.tokenizer.eos_token)) > 5:
                input_text = p.tokenizer.bos_token + f"{p.tokenizer.eos_token}".join(input_text.split(p.tokenizer.eos_token)[2:]) + p.tokenizer.eos_token

            output = model.generate(p.encoding(input_text),
                                    max_length=1000,
                                    num_beams=5,
                                    top_k=20,
                                    no_repeat_ngram_size=4,
                                    length_penalty=0.65,
                                    repetition_penalty=2.)
            bot_output = p.decoding(output)
            input_text += bot_output + p.tokenizer.eos_token

            print("HaYan >> " + bot_output)
            turn_text = input("User >> ")


