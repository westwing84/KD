# Knowledge Distillation(知識の蒸留)を用いてCIFAR10を小さいモデルに学習

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from Models import KDModel

# 定数宣言
NUM_CLASSES = 10        # 分類するクラス数
EPOCHS_T = 300          # Teacherモデルの学習回数
EPOCHS_S = 1000          # Studentモデルの学習回数
BATCH_SIZE = 512        # バッチサイズ
T = 2                   # 温度付きソフトマックスの温度
ALPHA = 0.5             # KD用のLossにおけるSoft Lossの割合
LR_T = 0.00001           # Teacherモデル学習時の学習率
LR_S = 0.001            # Studentモデル学習時の学習率


# F1-Scoreを求める関数
def f1_score(precision, recall):
    return (2 * precision * recall) / (precision + recall)


# set GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    # Create 4 virtual GPU
    try:
        i = 0
        tf.config.experimental.set_visible_devices(gpus[0:-1], 'GPU')
        for gpu in gpus[0:-1]:
            tf.config.experimental.set_memory_growth(gpu, True)
            i += 1

        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

cross_tower_ops = tf.distribute.HierarchicalCopyAllReduce(num_packs=3)
strategy = tf.distribute.MirroredStrategy(cross_device_ops=cross_tower_ops)

# config strategry
GLOBAL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync
print(GLOBAL_BATCH_SIZE)

# CIFAR10データセットの準備
(x, y), (x_test, y_test) = cifar10.load_data()
x = x.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y = to_categorical(y, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)
x = x.reshape([-1, 32, 32, 3])
x_test = x_test.reshape([-1, 32, 32, 3])

# MNISTのTrain用データをTrainとValidationに分割
validation_split = 0.2
idx_split = int(x.shape[0] * (1 - validation_split))
x_train, x_val = np.split(x, [idx_split])
y_train, y_val = np.split(y, [idx_split])
input_shape = x_train.shape[1:]

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(x_train.shape[0]).batch(BATCH_SIZE)
validation_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).shuffle(x_val.shape[0]).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(x_test.shape[0]).batch(BATCH_SIZE)
# uncertainty
mc = True
drop_rate = 0.5
n_ensemble = 20


# checkpoint
# checkpoint_prefix = "D:\\usr\\pras\\result\\arxiv\\KnowldegeDistillation\\"

# loss
# ---------------------------Epoch&Loss--------------------------#
loss_metric = tf.keras.metrics.Mean()
acc_metric = tf.keras.metrics.Mean()

test_loss_metric = tf.keras.metrics.Mean()
test_acc_metric = tf.keras.metrics.Mean()
# Loss function
with strategy.scope():
    # Set reduction to `none` so we can do the reduction afterwards and divide by

    # create model
    inputs = Input(shape=input_shape)
    teacher = KDModel.Teacher(NUM_CLASSES)
    student = KDModel.Students(NUM_CLASSES, T)
    model_T = teacher.createModel(inputs)
    model_S = student.createModel(inputs)

    # optimizer
    optimizer = Adam(learning_rate=LR_T)

    # checkpoint
    # checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model_T)
    # manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=3)

    # loss
    loss = tf.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE, from_logits=True)
    cross_loss = tf.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE, from_logits=True)
    acc = tf.keras.metrics.CategoricalAccuracy()


    def compute_loss(labels, predictions, global_batch_size):
        per_example_loss = loss(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)


    def compute_cross_loss(labels, predictions, global_batch_size):
        per_example_loss = cross_loss(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)


    def compute_acc(labels, predictions):
        return acc(labels, predictions)

with strategy.scope():
    def train_teacher(inputs):
        x_train = inputs[0]
        y_train = inputs[1]
        with tf.GradientTape() as tape:
            logits = model_T(x_train)
            probs = tf.nn.softmax(logits)
            loss = compute_loss(y_train, logits, GLOBAL_BATCH_SIZE)
            acc = compute_acc(y_train, probs)

        grads = tape.gradient(loss, model_T.trainable_variables)
        optimizer.apply_gradients(zip(grads, model_T.trainable_variables))

        loss_metric(loss)
        acc_metric(acc)
        return loss


    def test_teacher(inputs):
        x_test = inputs[0]
        y_test = inputs[1]
        logits_test = model_T(x_test)
        probs = tf.nn.softmax(logits_test)
        test_loss = loss(y_test, logits_test)
        test_acc = acc(y_test, probs)

        test_loss_metric(test_loss)
        test_acc_metric(test_acc)
        return test_loss


    def train_student(inputs):
        x_train = inputs[0]
        y_train = inputs[1]
        with tf.GradientTape() as tape:
            '''
            if mc:
                probs = []
                for i in range(n_ensemble):
                    probs.append(tf.expand_dims(tf.nn.softmax(model_T(x_train, training=True) / T), 0))
                probs_all_test = tf.concat(probs, axis=0)
                probs_mean = tf.reduce_mean(probs, 0)
                uncertainty = tf.reduce_mean(tf.reduce_sum((probs_all_test - probs_mean) ** 2, -1), 0)
                # uncertainty = tf.reduce_sum(tf.reduce_mean((probs_all_test - probs_mean) ** 2, 0), -1)
                logits = model_S(x_train)
                probs = tf.nn.softmax(logits)
                loss = (ALPHA * compute_loss(y_train, logits, GLOBAL_BATCH_SIZE)) + (
                        (uncertainty) * compute_cross_loss(probs_mean[0], logits / T, GLOBAL_BATCH_SIZE))

            else:
                teacher_pred = tf.nn.softmax(model_T(x_train) / T)
                logits = model_S(x_train)
                probs = tf.nn.softmax(logits)
                loss = ((1 - ALPHA) * compute_loss(y_train, logits, GLOBAL_BATCH_SIZE)) + (
                            ALPHA * compute_loss(teacher_pred, logits / T, GLOBAL_BATCH_SIZE))
            '''
            teacher_pred = tf.nn.softmax(model_T(x_train) / T)
            logits = model_S(x_train)
            probs = tf.nn.softmax(logits)
            loss = ((1 - ALPHA) * compute_loss(y_train, logits, GLOBAL_BATCH_SIZE)) + (
                    ALPHA * compute_loss(teacher_pred, logits / T, GLOBAL_BATCH_SIZE))
            acc = compute_acc(y_train, probs)

        grads = tape.gradient(loss, model_S.trainable_weights)
        optimizer.apply_gradients(zip(grads, model_S.trainable_weights))

        loss_metric(loss)
        acc_metric(acc)
        return loss


    def test_student(inputs):
        x_test = inputs[0]
        y_test = inputs[1]
        logits_test = model_S(x_test)
        probs = tf.nn.softmax(logits_test)
        test_loss = loss(y_test, logits_test)
        test_acc = acc(y_test, probs)

        test_loss_metric(test_loss)
        test_acc_metric(test_acc)

        return test_loss

with strategy.scope():
    # `experimental_run_v2` replicates the provided computation and runs it
    # with the distributed input.
    @tf.function
    def distributed_train_teacher(dataset_inputs):
        per_replica_losses = strategy.run(train_teacher, args=(dataset_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


    @tf.function
    def distributed_test_teacher(dataset_inputs):
        return strategy.run(test_teacher, args=(dataset_inputs,))


    @tf.function
    def distributed_train_student(dataset_inputs):
        per_replica_losses = strategy.run(train_student, args=(dataset_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


    @tf.function
    def distributed_test_student(dataset_inputs):
        return strategy.run(test_student, args=(dataset_inputs,))


    for epoch in range(EPOCHS_T):
        # TRAIN LOOP
        total_loss = 0.0
        num_batches = 0
        for step, train in enumerate(train_dataset):
            total_loss = distributed_train_teacher(train)
            # print(total_loss)

        # TEST LOOP
        for step_test, test in enumerate(test_dataset):
            distributed_test_teacher(test)


        template = ("Epoch {}, Loss: {}, ACC: {}, Test Loss: {}, Test ACC: {}")
        print(template.format(epoch + 1, loss_metric.result().numpy(), acc_metric.result().numpy(),
                                      test_loss_metric.result().numpy(), test_acc_metric.result().numpy()))
        loss_metric.reset_states()
        acc_metric.reset_states()
        test_loss_metric.reset_states()
        test_acc_metric.reset_states()
    # manager.save()

    print("--------------------Finish Training Teacher--------------------------")
    # checkpoint.restore(manager.latest_checkpoint)
    for epoch in range(EPOCHS_S):
        # TRAIN LOOP
        total_loss = 0.0
        num_batches = 0
        for step, train in enumerate(train_dataset):
            total_loss = distributed_train_student(train)
            # print(total_loss)

        # TEST LOOP
        for step_test, test in enumerate(test_dataset):
            distributed_test_student(test)

        template = ("Epoch {}, Loss: {}, ACC: {}, Test Loss: {}, "
                    "Test ACC: {}")
        print(template.format(epoch + 1, loss_metric.result().numpy(), acc_metric.result().numpy(),
                              test_loss_metric.result().numpy(), test_acc_metric.result().numpy()))

        loss_metric.reset_states()
        acc_metric.reset_states()
        test_loss_metric.reset_states()
        test_acc_metric.reset_states()

