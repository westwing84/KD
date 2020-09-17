# Knowledge Distillation(知識の蒸留)を用いてCIFAR10を小さいモデルに学習

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.metrics import Mean, CategoricalAccuracy, Precision, Recall
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils.vis_utils import plot_model
from Utils import LossAccHistory
import KDModel

# 定数宣言
NUM_CLASSES = 10        # 分類するクラス数
EPOCHS_T = 30            # Teacherモデルの学習回数
EPOCHS_S = 100           # Studentモデルの学習回数
BATCH_SIZE = 128        # バッチサイズ
VERBOSE = 2             # 学習進捗の表示モード
optimizer = Adam()      # 最適化アルゴリズム
T = 5                   # 温度付きソフトマックスの温度
ALPHA = 0.5             # KD用のLossにおけるSoft Lossの割合


# F1-Scoreを求める関数
def f1_score(precision, recall):
    return (2 * precision * recall) / (precision + recall)


# MNISTデータセットの準備
(x, y), (x_test, y_test) = cifar10.load_data()
y = to_categorical(y, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)
x = x.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x = x.reshape([-1, 32, 32, 3])
x_test = x_test.reshape([-1, 32, 32, 3])

# MNISTのTrain用データをTrainとValidationに分割
validation_split = 0.2
idx_split = int(x.shape[0] * (1 - validation_split))
x_train, x_val = np.split(x, [idx_split])
y_train, y_val = np.split(y, [idx_split])

# Trainデータを主データと補助データに分割．ValidationとTestデータは主データのみを残す．
x_train_main, x_train_aux = np.array_split(x_train, 2, axis=1)
x_val_main, x_val_aux = np.array_split(x_val, 2, axis=1)
x_test_main, x_test_aux = np.array_split(x_test, 2, axis=1)
input_shape_main = x_train_main.shape[1:]
input_shape_aux = x_train_aux.shape[1:]


# 入力データの表示
x_train_tmp = np.concatenate([x_train_main, 0*x_train_aux], axis=1)
x_test_tmp = np.concatenate([x_test_main, 0*x_test_aux], axis=1)
n = 10
plt.figure()
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_train_tmp[i].reshape(32, 32, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, n + i + 1)
    plt.imshow(x_test_tmp[i].reshape(32, 32, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# MNISTデータセットをtf.data.Datasetに変換
ds_train = tf.data.Dataset.from_tensor_slices((x_train_main, x_train_aux, y_train)).shuffle(x_train.shape[0]).batch(BATCH_SIZE)
ds_val = tf.data.Dataset.from_tensor_slices((x_val_main, x_val_aux, y_val)).shuffle(x_val.shape[0]).batch(BATCH_SIZE)
ds_test = tf.data.Dataset.from_tensor_slices((x_test_main, x_test_aux, y_test)).shuffle(x_test.shape[0]).batch(BATCH_SIZE)
ds = tf.data.Dataset.zip((ds_train, ds_val))

# Teacherモデルの定義
inputs_main = Input(shape=input_shape_main)
inputs_aux = Input(shape=input_shape_aux)
teacher = KDModel.Teacher(NUM_CLASSES, T)
teacher_model = teacher.createModel(inputs_main, inputs_aux)

# Teacherモデルの学習
training = KDModel.NormalTraining(teacher_model)
teacher_model.summary()
# plot_model(teacher_model, show_shapes=True, to_file='teacher_model.png')
for epoch in range(1, EPOCHS_T + 1):
    epoch_loss_avg = Mean()
    epoch_loss_avg_val = Mean()
    epoch_accuracy = CategoricalAccuracy()
    epoch_accuracy_val = CategoricalAccuracy()

    # 各バッチごとに学習
    for (x_train_main_, x_train_aux_, y_train_), (x_val_main_, x_val_aux_, y_val_) in ds:
        loss_value, grads = training.grad([x_train_main_, x_train_aux_], y_train_)
        optimizer.apply_gradients(zip(grads, teacher_model.trainable_variables))
        loss_value_test = training.loss([x_val_main_, x_val_aux_], y_val_)

        epoch_loss_avg(loss_value)
        epoch_accuracy(y_train_, teacher_model([x_train_main_, x_train_aux_]))
        epoch_loss_avg_val(loss_value_test)
        epoch_accuracy_val(y_val_, teacher_model([x_val_main_, x_val_aux_]))

    # 学習進捗の表示
    print('Epoch {}/{}: Loss: {:.3f}, Accuracy: {:.3%}, Validation Loss: {:.3f}, Validation Accuracy: {:.3%}'.format(
        epoch, EPOCHS_T, epoch_loss_avg.result(), epoch_accuracy.result(),
        epoch_loss_avg_val.result(), epoch_accuracy_val.result()))


# Studentモデルの定義
student = KDModel.Students(NUM_CLASSES, T)
student_hard_model = student.createHardModel(inputs_main)
student_soft_model = student.createSoftModel(inputs_main)

# Studentモデルの学習
student_hard_model.summary()
student_soft_model.summary()
# plot_model(student_soft_model, show_shapes=True, to_file='student_model.png')
kd = KDModel.KnowledgeDistillation(teacher_model, student_hard_model, student_soft_model, T, ALPHA)
history_student = LossAccHistory()
for epoch in range(1, EPOCHS_S + 1):
    epoch_loss_avg = Mean()
    epoch_loss_avg_val = Mean()
    epoch_accuracy = CategoricalAccuracy()
    epoch_accuracy_val = CategoricalAccuracy()

    # 各バッチごとに学習
    for (x_train_main_, x_train_aux_, y_train_), (x_val_main_, x_val_aux_, y_val_) in ds:
        loss_value, grads = kd.grad(x_train_main_, x_train_aux_, y_train_)
        optimizer.apply_gradients(zip(grads, student_hard_model.trainable_variables))
        loss_value_test = kd.loss(x_val_main_, x_val_aux_, y_val_)

        epoch_loss_avg(loss_value)
        epoch_accuracy(y_train_, student_hard_model(x_train_main_))
        epoch_loss_avg_val(loss_value_test)
        epoch_accuracy_val(y_val_, student_hard_model(x_val_main_))

    # 学習進捗の表示
    print('Epoch {}/{}: Loss: {:.3f}, Accuracy: {:.3%}, Validation Loss: {:.3f}, Validation Accuracy: {:.3%}'.format(
        epoch, EPOCHS_S, epoch_loss_avg.result(), epoch_accuracy.result(),
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
for (x_train_main_, x_train_aux_, y_train_), (x_val_main_, x_val_aux_, y_val_), (x_test_main_, x_test_aux_, y_test_) in ds:
    score_train[0](kd.loss(x_train_main_, x_train_aux_, y_train_))
    score_val[0](kd.loss(x_val_main_, x_val_aux_, y_val_))
    score_test[0](kd.loss(x_test_main_, x_test_aux_, y_test_))
    score_train[1](y_train_, student_hard_model(x_train_main_))
    score_val[1](y_val_, student_hard_model(x_val_main_))
    score_test[1](y_test_, student_hard_model(x_test_main_))
    score_train[2](y_train_, student_hard_model(x_train_main_))
    score_val[2](y_val_, student_hard_model(x_val_main_))
    score_test[2](y_test_, student_hard_model(x_test_main_))
    score_train[3](y_train_, student_hard_model(x_train_main_))
    score_val[3](y_val_, student_hard_model(x_val_main_))
    score_test[3](y_test_, student_hard_model(x_test_main_))
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

# LossとAccuracyをグラフにプロット
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


