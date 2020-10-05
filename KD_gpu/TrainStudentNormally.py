# Studentモデルを普通に学習させた場合の検証

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input
from tensorflow.keras.datasets import mnist
from tensorflow.keras.metrics import Mean, CategoricalAccuracy, Precision, Recall
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from Utils import LossAccHistory
from Models import KDModel

# 定数宣言
NUM_CLASSES = 10        # 分類するクラス数
EPOCHS = 100            # 学習回数
BATCH_SIZE = 128        # バッチサイズ
VALIDATION_SPLIT = 0.2  # 評価に用いるデータの割合
VERBOSE = 2             # 学習進捗の表示モード
optimizer = Adam()      # 最適化アルゴリズム


# F1-Scoreを求める関数
def f1_score(precision, recall):
    return (2 * precision * recall) / (precision + recall)


# MNISTデータセットの準備
(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = x_train.reshape([-1, 28, 28, 1])
x_test = x_test.reshape([-1, 28, 28, 1])

# MNISTのTrain用データをTrainとValidationに分割
idx_split = int(x_train.shape[0] * (1 - VALIDATION_SPLIT))
x_train, x_val = np.split(x_train, [idx_split])
y_train, y_val = np.split(y_train, [idx_split])

# Trainデータを主データと補助データに分割．ValidationとTestデータは主データのみを残す．
x_train_main, x_train_aux = np.array_split(x_train, 2, axis=1)
x_val_main, x_val_aux = np.array_split(x_val, 2, axis=1)
x_test_main, x_test_aux = np.array_split(x_test, 2, axis=1)
input_shape_main = x_train_main.shape[1:]
input_shape_aux = x_train_aux.shape[1:]

# MNISTデータセットをtf.data.Datasetに変換
ds_train = tf.data.Dataset.from_tensor_slices((x_train_main, y_train)).shuffle(x_train.shape[0]).batch(BATCH_SIZE)
ds_val = tf.data.Dataset.from_tensor_slices((x_val_main, y_val)).shuffle(x_val.shape[0]).batch(BATCH_SIZE)
ds_test = tf.data.Dataset.from_tensor_slices((x_test_main, y_test)).shuffle(x_test.shape[0]).batch(BATCH_SIZE)
ds = tf.data.Dataset.zip((ds_train, ds_val))

# Studentモデルの構築
inputs = Input(shape=input_shape_main)
student = KDModel.Students(NUM_CLASSES)
student_model = student.createHardModel(inputs)

# 学習
training = KDModel.NormalTraining(student_model)
history_student = LossAccHistory()
student_model.summary()
for epoch in range(1, EPOCHS + 1):
    epoch_loss_avg = Mean()
    epoch_loss_avg_val = Mean()
    epoch_accuracy = CategoricalAccuracy()
    epoch_accuracy_val = CategoricalAccuracy()

    # 各バッチごとに学習
    for (x_train_, y_train_), (x_val_, y_val_) in ds:
        loss_value, grads = training.grad(x_train_, y_train_)
        optimizer.apply_gradients(zip(grads, student_model.trainable_variables))
        loss_value_test = training.loss(x_val_, y_val_)

        epoch_loss_avg(loss_value)
        epoch_accuracy(y_train_, student_model(x_train_))
        epoch_loss_avg_val(loss_value_test)
        epoch_accuracy_val(y_val_, student_model(x_val_))

    # 学習進捗の表示
    print('Epoch {}/{}: Loss: {:.3f}, Accuracy: {:.3%}, Validation Loss: {:.3f}, Validation Accuracy: {:.3%}'.format(
        epoch, EPOCHS, epoch_loss_avg.result(), epoch_accuracy.result(),
        epoch_loss_avg_val.result(), epoch_accuracy_val.result()))
    # LossとAccuracyの記録(後でグラフにプロットするため)
    history_student.losses.append(epoch_loss_avg.result())
    history_student.accuracy.append(epoch_accuracy.result() * 100)
    history_student.losses_val.append(epoch_loss_avg_val.result())
    history_student.accuracy_val.append(epoch_accuracy_val.result() * 100)

# Studentモデルの評価
score_train = [Mean(), CategoricalAccuracy(), Precision(), Recall()]
score_val = [Mean(), CategoricalAccuracy(), Precision(), Recall()]
score_test = [Mean(), CategoricalAccuracy(), Precision(), Recall()]
ds = tf.data.Dataset.zip((ds_train, ds_val, ds_test))
for (x_train_, y_train_), (x_val_, y_val_), (x_test_, y_test_) in ds:
    score_train[0](training.loss(x_train_, y_train_))
    score_val[0](training.loss(x_val_, y_val_))
    score_test[0](training.loss(x_test_, y_test_))
    score_train[1](y_train_, student_model(x_train_))
    score_val[1](y_val_, student_model(x_val_))
    score_test[1](y_test_, student_model(x_test_))
    score_train[2](y_train_, student_model(x_train_))
    score_val[2](y_val_, student_model(x_val_))
    score_test[2](y_test_, student_model(x_test_))
    score_train[3](y_train_, student_model(x_train_))
    score_val[3](y_val_, student_model(x_val_))
    score_test[3](y_test_, student_model(x_test_))
f1_train = f1_score(score_train[2].result(), score_train[3].result())
f1_val = f1_score(score_val[2].result(), score_val[3].result())
f1_test = f1_score(score_test[2].result(), score_test[3].result())

print('-----------------------------Student Model------------------------------------')
print('Train - Loss: {:.3f}, Accuracy: {:.3%}, Precision: {:.3f}, Recall: {:.3f}, F1-Score: {:.3f}'.format(
    score_train[0].result(), score_train[1].result(), score_train[2].result(), score_train[3].result(), f1_train))
print('Validation - Loss: {:.3f}, Accuracy: {:.3%}, Precision: {:.3f}, Recall: {:.3f}, F1-Score: {:.3f}'.format(
    score_val[0].result(), score_val[1].result(), score_val[2].result(), score_val[3].result(), f1_val))
print('Test - Loss: {:.3f}, Accuracy: {:.3%}, Precision: {:.3f}, Recall: {:.3f}, F1-Score: {:.3f}'.format(
    score_test[0].result(), score_test[1].result(), score_test[2].result(), score_test[3].result(), f1_test))

# 損失と精度をグラフにプロット
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(history_student.accuracy)
plt.plot(history_student.accuracy_val)
plt.title('Student Model Accuracy')
plt.ylabel('Accuracy [%]')
plt.xlabel('Epoch')
plt.ylim(0.0, 101.0)
plt.legend(['Train', 'Validation'])

plt.subplot(1, 2, 2)
plt.plot(history_student.losses)
plt.plot(history_student.losses_val)
plt.title('Student Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.tight_layout()
plt.show()

