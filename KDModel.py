import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Lambda, concatenate
from tensorflow.keras.losses import CategoricalCrossentropy


# Teacherのモデル
class Teacher():

    def __init__(self, num_classes, temperature):
        self.num_classes = num_classes
        self.temperature = temperature

        self.conv1 = Conv2D(filters=16, kernel_size=(1, 1), name='conv1')
        self.conv2 = Conv2D(filters=32, kernel_size=(3, 3), name='conv2')
        self.conv3 = Conv2D(filters=64, kernel_size=(5, 5), name='conv3')
        self.maxpool = MaxPooling2D(pool_size=(2, 2))

        self.dense1 = Dense(units=512, name='dense1')
        self.dense2 = Dense(units=512, name='dense2')
        self.dense3 = Dense(units=num_classes, name='dense3')

        self.relu1 = Activation('relu', name='relu1')
        self.relu2 = Activation('relu', name='relu2')
        self.relu3 = Activation('relu', name='relu3')
        self.relu4 = Activation('relu', name='relu4')
        self.relu5 = Activation('relu', name='relu5')
        self.softmax = Activation('softmax', name='softmax')
        self.softmax_T = Activation('softmax', name='softmax_T')

        self.batch_norm1 = BatchNormalization(name='batch_norm1')
        self.batch_norm2 = BatchNormalization(name='batch_norm2')
        self.batch_norm3 = BatchNormalization(name='batch_norm3')
        self.batch_norm4 = BatchNormalization(name='batch_norm4')
        self.batch_norm5 = BatchNormalization(name='batch_norm5')

        self.flatten = Flatten()

        self.divide_T = Lambda(lambda x: x / temperature)

        self.model = None

    def createModel(self, inputs_main, inputs_aux):
        x = concatenate([inputs_main, inputs_aux], axis=1)
        x = self.relu1(self.batch_norm1(self.conv1(x)))
        x = self.relu2(self.batch_norm2(self.conv2(x)))
        x = self.relu3(self.batch_norm3(self.maxpool(self.conv3(x))))
        x = self.flatten(x)
        x = self.relu4(self.batch_norm4(self.dense1(x)))
        x = self.relu5(self.batch_norm5(self.dense2(x)))
        outputs = self.softmax(self.dense3(x))
        model = Model([inputs_main, inputs_aux], outputs, name='TeacherModel')
        self.model = model

        return model

    def recreateModel(self):
        self.model.layers.pop()
        inputs = self.model.input
        x = self.model.layers[-1].output
        x = self.divide_T(x)
        outputs = self.softmax_T(x)
        model = Model(inputs, outputs, name='TeacherModel')

        return model


# Studentのモデル
class Students():

    def __init__(self, num_classes, temperature=5):
        self.num_classes = num_classes
        self.temperature = temperature

        self.dense1 = Dense(units=100, name='dense1')
        self.dense2 = Dense(units=100, name='dense2')
        self.dense3 = Dense(units=num_classes, name='dense3')

        self.relu1 = Activation('relu', name='relu1')
        self.relu2 = Activation('relu', name='relu2')
        self.softmax = Activation('softmax', name='softmax')
        self.softmax_T = Activation('softmax', name='softmax_T')

        self.batch_norm1 = BatchNormalization(name='batch_norm1')
        self.batch_norm2 = BatchNormalization(name='batch_norm2')

        self.flatten = Flatten()

        self.divide_T = Lambda(lambda x: x / temperature)

        self.logits = None

    def createHardModel(self, inputs):
        x = self.flatten(inputs)
        x = self.relu1(self.batch_norm1(self.dense1(x)))
        x = self.relu2(self.batch_norm2(self.dense2(x)))
        logits = self.dense3(x)
        self.logits = logits
        outputs = self.softmax(logits)
        hard_model = Model(inputs, outputs, name='StudentHardModel')

        return hard_model

    def createSoftModel(self, inputs):
        logits_T = self.divide_T(self.logits)
        outputs = self.softmax_T(logits_T)
        soft_model = Model(inputs, outputs, name='StudentSoftModel')

        return soft_model


# Knowledge DistillationのLossおよび勾配計算の定義
class KnowledgeDistillation():

    def __init__(self, teacher_model, student_hard_model, student_soft_model, temperature, alpha):
        self.teacher_model = teacher_model
        self.student_hard_model = student_hard_model
        self.student_soft_model = student_soft_model
        self.temperature = temperature
        self.alpha = alpha

    @tf.function
    def loss(self, x_main, x_aux, y_true):
        loss_object = CategoricalCrossentropy()
        yt_soft = self.teacher_model([x_main, x_aux])
        ys_soft = self.student_soft_model(x_main)
        ys_hard = self.student_hard_model(x_main)
        loss_value = (1 - self.alpha) * loss_object(y_true, ys_hard) + \
                      self.alpha * (self.temperature ** 2) * loss_object(yt_soft, ys_soft)

        # loss_value = (1 - self.alpha) * loss_object(y_true, ys_hard) + self.alpha * loss_object(yt_soft, ys_soft)
        return loss_value

    @tf.function
    def grad(self, x_main, x_aux, targets):
        with tf.GradientTape() as tape:
            loss_value = self.loss(x_main, x_aux, targets)
        return loss_value, tape.gradient(loss_value, self.student_hard_model.trainable_variables)


# 通常の学習のためのLoss, Grad
class NormalTraining():
    def __init__(self, model: Model):
        self.model = model

    @tf.function
    def loss(self, x, y_true):
        loss_object = CategoricalCrossentropy()
        y_pred = self.model(x)
        return loss_object(y_true=y_true, y_pred=y_pred)

    @tf.function
    def grad(self, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self.loss(inputs, targets)
        return loss_value, tape.gradient(loss_value, self.model.trainable_variables)
