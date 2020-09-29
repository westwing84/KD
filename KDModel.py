import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Dropout, Conv2D, MaxPooling2D,\
    GlobalAveragePooling2D, Flatten, Lambda, concatenate
from tensorflow.keras.losses import CategoricalCrossentropy


# Teacherのモデル
class Teacher():

    def __init__(self, num_classes, temperature=5):
        self.num_classes = num_classes
        self.temperature = temperature

    def createModel(self, inputs_main, inputs_aux=None):
        if inputs_aux == None:
            x = inputs_main
        else:
            x = concatenate([inputs_main, inputs_aux], axis=1)
        x = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = Activation('relu')(BatchNormalization()(x))
        x = Conv2D(32, (5, 5), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.5)(x)

        x = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = Activation('relu')(BatchNormalization()(x))
        x = Conv2D(64, (5, 5), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.5)(x)

        x = Conv2D(128, (1, 1), padding='same', activation='relu')(x)
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = Activation('relu')(BatchNormalization()(x))
        x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(128, (5, 5), padding='same', activation='relu')(x)
        x = GlobalAveragePooling2D()(x)

        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        logits = Dense(self.num_classes)(x)

        if inputs_aux == None:
            model = Model(inputs_main, logits, name='TeacherModel')
        else:
            model = Model([inputs_main, inputs_aux], logits, name='TeacherModel')

        return model


# Studentのモデル
class Students():

    def __init__(self, num_classes, temperature=5):
        self.num_classes = num_classes
        self.temperature = temperature

    def createModel(self, inputs):
        x = Conv2D(8, (1, 1), padding='same')(inputs)
        x = Activation('relu')(BatchNormalization()(x))
        x = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.5)(x)

        x = Conv2D(16, (1, 1), padding='same')(x)
        x = Activation('relu')(BatchNormalization()(x))
        x = Conv2D(16, (3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.5)(x)

        x = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
        x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = GlobalAveragePooling2D()(x)

        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        logits = Dense(self.num_classes)(x)

        model = Model(inputs, logits, name='StudentModel')

        return model


# Knowledge DistillationのLossおよび勾配計算の定義
class KnowledgeDistillation():

    def __init__(self, teacher_model, student_model, temperature, alpha):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha

    @tf.function
    def loss(self, x, y_true):
        loss_object = CategoricalCrossentropy()
        yt_soft = tf.nn.softmax(self.teacher_model(x) / self.temperature)
        ys_soft = tf.nn.softmax(self.student_model(x) / self.temperature)
        ys_hard = tf.nn.softmax(self.student_model(x))
        loss_value = (1 - self.alpha) * loss_object(y_true, ys_hard) + \
                     self.alpha * (self.temperature ** 2) * loss_object(yt_soft, ys_soft)
        return loss_value

    @tf.function
    def grad(self, x, targets):
        with tf.GradientTape() as tape:
            loss_value = self.loss(x, targets)
        return loss_value, tape.gradient(loss_value, self.student_model.trainable_variables)

    @tf.function
    def loss_mainaux(self, x_main, x_aux, y_true):
        loss_object = CategoricalCrossentropy()
        yt_soft = tf.nn.softmax(self.teacher_model([x_main, x_aux]) / self.temperature)
        ys_soft = tf.nn.softmax(self.student_model(x_main) / self.temperature)
        ys_hard = tf.nn.softmax(self.student_model(x_main))
        loss_value = (1 - self.alpha) * loss_object(y_true, ys_hard) + \
                      self.alpha * loss_object(yt_soft, ys_soft)

        # loss_value = (1 - self.alpha) * loss_object(y_true, ys_hard) + self.alpha * loss_object(yt_soft, ys_soft)
        return loss_value

    @tf.function
    def grad_mainaux(self, x_main, x_aux, targets):
        with tf.GradientTape() as tape:
            loss_value = self.loss_mainaux(x_main, x_aux, targets)
        return loss_value, tape.gradient(loss_value, self.student_model.trainable_variables)


# 通常の学習のためのLoss, Grad
class NormalTraining():
    def __init__(self, model: Model):
        self.model = model

    @tf.function
    def loss(self, x, y_true):
        loss_object = CategoricalCrossentropy()
        y_pred = tf.nn.softmax(self.model(x))
        return loss_object(y_true=y_true, y_pred=y_pred)

    @tf.function
    def grad(self, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self.loss(inputs, targets)
        return loss_value, tape.gradient(loss_value, self.model.trainable_variables)
