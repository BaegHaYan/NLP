from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from transformers import TFGPT2LMHeadModel
from HaYan_NLP.preprocessing import Preprocesser
import tensorflow as tf

def lr_scheduler(epoch, lr):
    if epoch < 2:
        return 5e-5
    elif epoch < 4:
        return 3e-5
    else:
        return 1e-5

class DialoGPT(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(DialoGPT, self).__init__(*args, **kwargs)
        self.koDialoGPT = TFGPT2LMHeadModel.from_pretrained(p.PREMODEL_NAME, from_pt=True)

    def call(self, inputs, training=None, mask=None):
        # {'input_ids': (batch, max_len), 'attention_mask': (batch, max_len)
        # -> (batch, max_len, vocab_size(51200))
        output = self.koDialoGPT(inputs, return_dict=True)
        return output.logits


if __name__ == "__main__":
    p = Preprocesser()
    epochs = 5

    model = DialoGPT()
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    history = ""
    for optim in ["adam", "rmsprop", "nadam"]:
        model.compile(loss=loss, optimizer=optim, metrics="accuracy")
        hist = model.fit(p.getTrainData(), validation_data=p.getValidationData(), batch_size=p.batch_size, epochs=epochs,
                         callbacks=[EarlyStopping(monitor='val_loss', mode="min", patience=5), LearningRateScheduler(lr_scheduler),
                                    ModelCheckpoint("./model/"+optim+"_model", monitor="val_accuracy", save_best_only=True)])  # have to tf_model.h5
        history += optim+"\n"
        for key, item in hist.history.items():
            history += key + " : " + str(["%.4f" % figure for figure in item]) + "\n"
        history += "\n"
    open("../model/history.txt", "w+", encoding="utf-8").write(history)

