from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation
import tensorflow as tf


def createVGG16(inputs, num_classes):

    x = RandomFlip()(inputs)
    x = RandomRotation(0.2)(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(512, (3, 3), padding='same')(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(512, (3, 3), padding='same')(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(4096)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096)(x)
    x = Activation('relu')(x)
    logits = Dense(num_classes)(x)

    return logits


'''
class VGG16(tf.keras.Model):

    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.num_classes = num_classes
        
        # data augmentation
        self.flip = tf.keras.layers.experimental.preprocessing.RandomFlip()
        self.rotation = tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)

        # Convolution
        self.conv64_1 = Conv2D(64, (3, 3), padding='same')
        self.conv64_2 = Conv2D(64, (3, 3), padding='same')
        self.conv128_1 = Conv2D(128, (3, 3), padding='same')
        self.conv128_2 = Conv2D(128, (3, 3), padding='same')
        self.conv256_1 = Conv2D(256, (3, 3), padding='same')
        self.conv256_2 = Conv2D(256, (3, 3), padding='same')
        self.conv256_3 = Conv2D(256, (3, 3), padding='same')
        self.conv512_1 = Conv2D(512, (3, 3), padding='same')
        self.conv512_2 = Conv2D(512, (3, 3), padding='same')
        self.conv512_3 = Conv2D(512, (3, 3), padding='same')
        self.conv512_4 = Conv2D(512, (3, 3), padding='same')
        self.conv512_5 = Conv2D(512, (3, 3), padding='same')
        self.conv512_6 = Conv2D(512, (3, 3), padding='same')

        # Max Pooling
        self.maxpool1 = MaxPooling2D((2, 2))
        self.maxpool2 = MaxPooling2D((2, 2))
        self.maxpool3 = MaxPooling2D((2, 2))
        self.maxpool4 = MaxPooling2D((2, 2))
        self.maxpool5 = MaxPooling2D((2, 2))

        # Dense
        self.dense4096_1 = Dense(4096)
        self.dense4096_2 = Dense(4096)
        self.dense_out = Dense(num_classes)

        # Batch Normalization
        self.batch_norm1_1 = BatchNormalization()
        self.batch_norm1_2 = BatchNormalization()
        self.batch_norm2_1 = BatchNormalization()
        self.batch_norm2_2 = BatchNormalization()
        self.batch_norm3_1 = BatchNormalization()
        self.batch_norm3_2 = BatchNormalization()
        self.batch_norm3_3 = BatchNormalization()
        self.batch_norm4_1 = BatchNormalization()
        self.batch_norm4_2 = BatchNormalization()
        self.batch_norm4_3 = BatchNormalization()
        self.batch_norm5_1 = BatchNormalization()
        self.batch_norm5_2 = BatchNormalization()
        self.batch_norm5_3 = BatchNormalization()

        # Dropout
        self.dropout1 = Dropout(0.5)
        self.dropout2 = Dropout(0.5)

        # Activation
        self.relu = Activation('relu')


    def call(self, inputs, training=True, mask=None):
        x = self.relu(self.batch_norm1_1(self.conv64_1(inputs)))
        x = self.relu(self.batch_norm1_2(self.conv64_2(x)))
        x = self.maxpool1(x)

        x = self.relu(self.batch_norm2_1(self.conv128_1(x)))
        x = self.relu(self.batch_norm2_2(self.conv128_2(x)))
        x = self.maxpool2(x)

        return x
'''
