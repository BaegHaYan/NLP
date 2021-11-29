from preprocessing import Preprocesser
import tensorflow as tf
import sys

if __name__ == "__main__":
    p = Preprocesser()
    model = tf.keras.models.load_model("./model/tf_model")
    try:
        text = sys.argv[1]
    except IndexError:
        text = input("User >> ")

    output = model.generate(p.encoding(text),
                            max_length=1000,
                            num_beams=5,
                            top_k=20,
                            no_repeat_ngram_size=4,
                            length_penalty=0.65,
                            repetition_penalty=2.
                            )
    print(p.decoding(output))
