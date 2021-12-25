from preprocessing import Preprocesser
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--once", type=bool, default=False, metavar="Bool", dest="is_once")

if __name__ == "__main__":
    p = Preprocesser()
    model = None

    if parser.is_once:
        parser.add_argument('-text', '--text', type=str, default=input("User >>"), metavar='str', dest="text",
                            help='condition about using HF model')
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
        text = input("User >> ")
        while "quit" not in text:
            # text검사 -> 모델사용 -> 입력
            pass
