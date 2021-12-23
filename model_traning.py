from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from transformers import TFGPT2LMHeadModel
from preprocessing import Preprocesser
import matplotlib.pyplot as plt
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

    def get_config(self):
        return self.koDialoGPT.config


if __name__ == "__main__":
    p = Preprocesser()
    epochs = 5

    pos = 1
    model = DialoGPT()
    loss = tf.keras.losses.SparseCategoricalCrossentropy()

    for optim in ["adam", "rmsprop", "nadam"]:
        plt.subplot(3, 1, pos)
        plt.figure(figsize=(50, 50))
        pos += 1
        model.compile(loss=loss, optimizer=optim, metrics="accuracy")
        hist = model.fit(p.getTrainData(), validation_data=p.getValidationData(), batch_size=p.batch_size, epochs=epochs,
                         callbacks=[EarlyStopping(monitor='val_loss', mode="min", patience=5), LearningRateScheduler(lr_scheduler),
                                    ModelCheckpoint("./model/"+optim+"_model", monitor="val_accuracy", save_best_only=True)])  # tf_model.h5

        plt.plot(range(1, len(hist.history["loss"])+1), hist.history["loss"], "r", label="loss")
        plt.plot(range(1, len(hist.history["loss"])+1), hist.history["accuracy"], "b", label="accuracy")
        plt.plot(range(1, len(hist.history["loss"])+1), hist.history["val_loss"], "g", label="val_loss")
        plt.plot(range(1, len(hist.history["loss"])+1), hist.history["val_accuracy"], "k", label="val_accuracy")
        plt.title(optim)
        plt.text(1, 3, "max val_acc : %.1f" % max(hist.history["val_accuracy"]))
        plt.xlabel("epoch")
        plt.ylabel("loss/accuracy")
        plt.xticks(range(1, len(hist.history["loss"])+1))
        plt.xlim(0.9, len(hist.history["loss"])+0.1)
    plt.legend()
    plt.savefig("./history.png")
