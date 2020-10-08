from tensorflow.keras.callbacks import Callback


class LossAccHistory(Callback):
    def __init__(self):
        self.losses = []
        self.accuracy = []
        self.losses_val = []
        self.accuracy_val = []

    def on_epoch_end(self, epoch, logs={}):
        loss = logs.get('loss')
        accuracy = logs.get('accuracy') * 100
        val_loss = logs.get('val_loss')
        val_accuracy = logs.get('val_accuracy') * 100
        self.losses.append(loss)
        self.accuracy.append(accuracy)
        self.losses_val.append(val_loss)
        self.accuracy_val.append(val_accuracy)
