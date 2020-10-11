from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dropout


def createVGG16(inputs, num_classes):

    x = Conv2D(64, (3, 3), padding='same')(inputs)
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
    x = Dense(4096)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    logits = Dense(num_classes)(x)

    return logits

