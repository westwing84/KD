
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Dropout, Conv2D, MaxPooling2D,\
    GlobalAveragePooling2D, SeparableConv2D, add


def xceptionModel(inputs, num_classes):

    # entry flow
    x = Conv2D(32, (3, 3), strides=2)(inputs)
    x = Activation('relu')(BatchNormalization()(x))
    x = Conv2D(64, (3, 3), strides=2)(x)
    x = Activation('relu')(BatchNormalization()(x))

    residual = Conv2D(128, (1, 1), strides=2, padding='same')(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3), padding='same')(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = SeparableConv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)
    x = add([x, residual])

    residual = Conv2D(256, (1, 1), strides=2, padding='same')(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(256, (3, 3), padding='same')(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = SeparableConv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)
    x = add([x, residual])

    residual = Conv2D(728, (1, 1), strides=2, padding='same')(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(728, (3, 3), padding='same')(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = SeparableConv2D(728, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)
    x = add([x, residual])

    # middle flow
    for i in range(8):
        residual = x
        x = Activation('relu')(x)
        x = SeparableConv2D(728, (3, 3), padding='same')(x)
        x = Activation('relu')(BatchNormalization()(x))
        x = SeparableConv2D(728, (3, 3), padding='same')(x)
        x = Activation('relu')(BatchNormalization()(x))
        x = SeparableConv2D(728, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = add([x, residual])

    # exit flow
    residual = Conv2D(1024, (1, 1), strides=2, padding='same')(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu')(x)
    x = SeparableConv2D(728, (3, 3), padding='same')(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = SeparableConv2D(1024, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)
    x = add([x, residual])

    x = SeparableConv2D(1536, (3, 3), padding='same')(x)
    x = Activation('relu')(BatchNormalization()(x))
    x = SeparableConv2D(2048, (3, 3), padding='same')(x)
    x = Activation('relu')(BatchNormalization()(x))

    x = GlobalAveragePooling2D()(x)
    logits = Dense(num_classes, kernel_initializer='he_normal')(x)

    model = Model(inputs, logits)
    return model

