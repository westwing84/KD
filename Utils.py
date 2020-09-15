from tensorflow.keras.callbacks import Callback


class LossAccHistory(Callback):
    def __init__(self):
        self.losses = []
        self.accuracy = []
        self.losses_val = []
        self.accuracy_val = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('accuracy'))
        self.losses_val.append(logs.get('val_loss'))
        self.accuracy_val.append(logs.get('val_accuracy'))
