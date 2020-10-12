# Knowledge Distillation(知識の蒸留)を用いてCIFAR10を小さいモデルに学習

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Mean, CategoricalAccuracy, Precision, Recall
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils.vis_utils import plot_model
from Utils import LossAccHistory
from Models import KDModel

# 定数宣言
NUM_CLASSES = 10        # 分類するクラス数
EPOCHS_T = 100          # Teacherモデルの学習回数
EPOCHS_S = 1000          # Studentモデルの学習回数
BATCH_SIZE_T = 128      # Teacherモデル学習時のバッチサイズ
BATCH_SIZE_S = 512     # Studentモデル学習時のバッチサイズ
T = 2                   # 温度付きソフトマックスの温度
ALPHA = 0.5             # KD用のLossにおけるSoft Lossの割合
LR_T = 0.0001           # Teacherモデル学習時の学習率
LR_S = 0.001            # Studentモデル学習時の学習率


# F1-Scoreを求める関数
def f1_score(precision, recall):
    return (2 * precision * recall) / (precision + recall)


# CIFAR10データセットの準備
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
input_shape = x_train.shape[1:]


'''
# 入力データの表示
n = 10
plt.figure()
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_train[i].reshape(32, 32, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, n + i + 1)
    plt.imshow(x_test[i].reshape(32, 32, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
'''

# CIFAR-10をデータセット化し，Train, Validation, Testに分割
ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(x_train.shape[0]).batch(BATCH_SIZE_T)
ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).shuffle(x_val.shape[0]).batch(BATCH_SIZE_T)
ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(x_test.shape[0]).batch(BATCH_SIZE_T)
ds = tf.data.Dataset.zip((ds_train, ds_val))

# Teacherモデルの定義
inputs = Input(shape=input_shape)
teacher = KDModel.Teacher(NUM_CLASSES)
teacher_model = teacher.createModel(inputs)

# Teacherモデルの学習
optimizer = Adam(learning_rate=LR_T)      # 最適化アルゴリズム
history_teacher = LossAccHistory()

'''
teacher_model.compile(optimizer=optimizer,
                      loss=CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
teacher_model.summary()
teacher_model.fit(x_train, y_train,
                  batch_size=BATCH_SIZE_T,
                  epochs=EPOCHS_T,
                  verbose=2,
                  validation_data=(x_val, y_val),
                  callbacks=[history_teacher])
'''

training = KDModel.NormalTraining(teacher_model)
teacher_model.summary()
plot_model(teacher_model, show_shapes=True, to_file='teacher_model.png')
for epoch in range(1, EPOCHS_T + 1):
    loss_metric = Mean()
    loss_metric_val = Mean()
    acc_metric = Mean()
    acc_metric_val = Mean()
    acc = CategoricalAccuracy()
    acc_val = CategoricalAccuracy()

    # 各バッチごとに学習
    for (x_train_, y_train_), (x_val_, y_val_) in ds:
        loss_value, grads = training.grad(x_train_, y_train_)
        optimizer.apply_gradients(zip(grads, teacher_model.trainable_variables))
        loss_value_test = training.loss(x_val_, y_val_)
        probs = tf.nn.softmax(teacher_model(x_train_))
        probs_val = tf.nn.softmax(teacher_model(x_val_))

        loss_metric(loss_value)
        acc_metric(acc(y_train_, probs))
        loss_metric_val(loss_value_test)
        acc_metric_val(acc_val(y_val_, probs_val))

    # 学習進捗の表示
    print('Epoch {}/{}: Loss: {:.3f}, Accuracy: {:.3%}, Validation Loss: {:.3f}, Validation Accuracy: {:.3%}'.format(
        epoch, EPOCHS_T, loss_metric.result().numpy(), acc_metric.result().numpy(),
        loss_metric_val.result().numpy(), acc_metric_val.result().numpy()))
    # LossとAccuracyの記録(後でグラフにプロットするため)
    history_teacher.losses.append(loss_metric.result().numpy())
    history_teacher.accuracy.append(acc_metric.result().numpy() * 100)
    history_teacher.losses_val.append(loss_metric_val.result().numpy())
    history_teacher.accuracy_val.append(acc_metric_val.result().numpy() * 100)


# バッチサイズ変更
ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(x_train.shape[0]).batch(BATCH_SIZE_S)
ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).shuffle(x_val.shape[0]).batch(BATCH_SIZE_S)
ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(x_test.shape[0]).batch(BATCH_SIZE_S)
ds = tf.data.Dataset.zip((ds_train, ds_val))

# Studentモデルの定義
student = KDModel.Students(NUM_CLASSES, T)
student_model = student.createModel(inputs)

# Studentモデルの学習
optimizer = Adam(learning_rate=LR_S)      # 最適化アルゴリズム
student_model.summary()
# plot_model(student_model, show_shapes=True, to_file='student_model.png')
kd = KDModel.KnowledgeDistillation(teacher_model, student_model, T, ALPHA)
history_student = LossAccHistory()
for epoch in range(1, EPOCHS_S + 1):
    loss_metric = Mean()
    loss_metric_val = Mean()
    acc_metric = Mean()
    acc_metric_val = Mean()
    acc = CategoricalAccuracy()
    acc_val = CategoricalAccuracy()

    # 各バッチごとに学習
    for (x_train_, y_train_), (x_val_, y_val_) in ds:
        loss_value, grads = kd.grad(x_train_, y_train_)
        optimizer.apply_gradients(zip(grads, student_model.trainable_variables))
        loss_value_test = kd.loss(x_val_, y_val_)
        probs = tf.nn.softmax(student_model(x_train_))
        probs_val = tf.nn.softmax(student_model(x_val_))

        loss_metric(loss_value)
        acc_metric(acc(y_train_, probs))
        loss_metric_val(loss_value_test)
        acc_metric_val(acc_val(y_val_, probs_val))

    # 学習進捗の表示
    print('Epoch {}/{}: Loss: {:.3f}, Accuracy: {:.3%}, Validation Loss: {:.3f}, Validation Accuracy: {:.3%}'.format(
        epoch, EPOCHS_S, loss_metric.result().numpy(), acc_metric.result().numpy(),
        loss_metric_val.result().numpy(), acc_metric_val.result().numpy()))
    # LossとAccuracyの記録(後でグラフにプロットするため)
    history_student.losses.append(loss_metric.result().numpy())
    history_student.accuracy.append(acc_metric.result().numpy() * 100)
    history_student.losses_val.append(loss_metric_val.result().numpy())
    history_student.accuracy_val.append(acc_metric_val.result().numpy() * 100)

# Studentモデルの評価
score_train = [Mean(), CategoricalAccuracy(), Precision(), Recall()]
score_val = [Mean(), CategoricalAccuracy(), Precision(), Recall()]
score_test = [Mean(), CategoricalAccuracy(), Precision(), Recall()]
ds = tf.data.Dataset.zip((ds_train, ds_val, ds_test))
for (x_train_, y_train_), (x_val_, y_val_), (x_test_, y_test_) in ds:
    probs_train = tf.nn.softmax(student_model(x_train_))
    probs_val = tf.nn.softmax(student_model(x_val_))
    probs_test = tf.nn.softmax(student_model(x_test_))
    score_train[0](kd.loss(x_train_, y_train_))
    score_val[0](kd.loss(x_val_, y_val_))
    score_test[0](kd.loss(x_test_, y_test_))
    score_train[1](y_train_, probs_train)
    score_val[1](y_val_, probs_val)
    score_test[1](y_test_, probs_test)
    score_train[2](y_train_, probs_train)
    score_val[2](y_val_, probs_val)
    score_test[2](y_test_, probs_test)
    score_train[3](y_train_, probs_train)
    score_val[3](y_val_, probs_val)
    score_test[3](y_test_, probs_test)
f1_train = f1_score(score_train[2].result().numpy(), score_train[3].result().numpy())
f1_val = f1_score(score_val[2].result().numpy(), score_val[3].result().numpy())
f1_test = f1_score(score_test[2].result().numpy(), score_test[3].result().numpy())

print('-----------------------------Student Model------------------------------------')
print('Train - Loss: {:.3f}, Accuracy: {:.3%}, Precision: {:.3f}, Recall: {:.3f}, F1-Score: {:.3f}'.format(
    score_train[0].result().numpy(), score_train[1].result().numpy(), score_train[2].result().numpy(), score_train[3].result().numpy(), f1_train))
print('Validation - Loss: {:.3f}, Accuracy: {:.3%}, Precision: {:.3f}, Recall: {:.3f}, F1-Score: {:.3f}'.format(
    score_val[0].result().numpy(), score_val[1].result().numpy(), score_val[2].result().numpy(), score_val[3].result().numpy(), f1_val))
print('Test - Loss: {:.3f}, Accuracy: {:.3%}, Precision: {:.3f}, Recall: {:.3f}, F1-Score: {:.3f}'.format(
    score_test[0].result().numpy(), score_test[1].result().numpy(), score_test[2].result().numpy(), score_test[3].result().numpy(), f1_test))

# LossとAccuracyをグラフにプロット
# Teacherモデルの学習結果
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(history_teacher.accuracy)
plt.plot(history_teacher.accuracy_val)
plt.title('Teacher Model Accuracy')
plt.ylabel('Accuracy [%]')
plt.xlabel('Epoch')
plt.ylim(0.0, 101.0)
plt.legend(['Train', 'Validation'])

plt.subplot(1, 2, 2)
plt.plot(history_teacher.losses)
plt.plot(history_teacher.losses_val)
plt.title('Teacher Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.ylim(0.0, 5.0)
plt.legend(['Train', 'Validation'])
plt.tight_layout()

# Studentモデルの学習結果
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


