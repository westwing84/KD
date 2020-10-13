import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Dropout, Conv2D, MaxPooling2D,\
    Flatten, concatenate
from tensorflow.keras.losses import CategoricalCrossentropy, KLDivergence
from Models.Xception import createXception
from Models.VGG16 import createVGG16


# Teacherのモデル
class Teacher():

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def createModel(self, inputs_main, inputs_aux=None):
        if inputs_aux == None:
            x = inputs_main
        else:
            x = concatenate([inputs_main, inputs_aux], axis=1)
        logits = createVGG16(x, self.num_classes)
        # probs = Activation('softmax')(logits)

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
        x = Conv2D(8, (3, 3), padding='same')(inputs)
        x = Activation('relu')(BatchNormalization()(x))
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(8, (3, 3), padding='same')(x)
        x = Activation('relu')(BatchNormalization()(x))
        x = Dropout(0.25)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.5)(x)
        logits = Dense(self.num_classes)(x)

        model = Model(inputs, logits, name='StudentModel')

        return model


# Knowledge DistillationのLossおよび勾配計算の定義
class KnowledgeDistillation():

    def __init__(self, teacher_model: Model, student_model: Model, temperature, alpha):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha

    @tf.function
    def loss(self, x, y_true):
        loss_object = CategoricalCrossentropy(from_logits=True)
        loss_dist = KLDivergence()
        teacher_pred = tf.nn.softmax(self.teacher_model(x) / self.temperature)
        logits = self.student_model(x)
        loss_value = ((1 - self.alpha) * loss_object(y_true, logits)) + \
                     (self.alpha * loss_dist(teacher_pred, tf.nn.softmax(logits / self.temperature)))
        return loss_value

    @tf.function
    def grad(self, x, targets):
        with tf.GradientTape() as tape:
            loss_value = self.loss(x, targets)
        return loss_value, tape.gradient(loss_value, self.student_model.trainable_weights)

    @tf.function
    def loss_mainaux(self, x_main, x_aux, y_true):
        loss_object = CategoricalCrossentropy(from_logits=True)
        teacher_pred = tf.nn.softmax(self.teacher_model([x_main, x_aux]))
        logits = self.student_model(x_main)
        loss_value = (1 - self.alpha) * loss_object(y_true, logits) + \
                      self.alpha * loss_object(teacher_pred, logits / self.temperature)
        return loss_value

    @tf.function
    def grad_mainaux(self, x_main, x_aux, targets):
        with tf.GradientTape() as tape:
            loss_value = self.loss_mainaux(x_main, x_aux, targets)
        return loss_value, tape.gradient(loss_value, self.student_model.trainable_weights)


# 通常の学習のためのLoss, Grad
class NormalTraining():
    def __init__(self, model: Model):
        self.model = model

    @tf.function
    def loss(self, x, y_true):
        loss_object = CategoricalCrossentropy(from_logits=True)
        logits = self.model(x)
        return loss_object(y_true, logits)

    @tf.function
    def grad(self, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self.loss(inputs, targets)
        return loss_value, tape.gradient(loss_value, self.model.trainable_weights)
