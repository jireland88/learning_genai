import numpy as np
from tensorflow.keras import datasets, utils, layers, models, optimizers

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

    # First scale the pixel values from [0, 255] to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Next One-Hot Code the output
    n_classes = len(np.unique(y_train))
    y_train = utils.to_categorical(y_train, n_classes)
    y_test = utils.to_categorical(y_test, n_classes)

    # Build the Neural Network    
    input_layer = layers.Input(shape=(32, 32, 3))
    x = layers.Flatten()(input_layer)
    x = layers.Dense(units=200, activation="relu")(x)
    x = layers.Dense(units=150, activation="relu")(x)
    output_layer = layers.Dense(units=10, activation="softmax")(x)
    
    model = models.Model(input_layer, output_layer)

    opt = optimizers.Adam(learning_rate=0.0005)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    model.fit(x_train, y_train, batch_size=32, epochs=10, shuffle=True)

    model.evaluate(x_test, y_test)

    # Improve to a CNN
    input_layer = layers.Input(shape=(32, 32, 3))
    conv_layer_1 = layers.Conv2D(
            filters=10,
            kernel_size=(4,4),
            strides=2,
            padding="same"
            )(input_layer)
    conv_layer_2 = layers.Conv2D(
            filters=20,
            kernel_size=(3,3),
            strides=2,
            padding="same",
            )(conv_layer_1)
    flatten_layer=layers.Flatten()(conv_layer_1)
    output_layer=layers.Dense(units=10, activation="softmax")(flatten_layer)
    model = models.Model(input_layer, output_layer)

    opt = optimizers.Adam(learning_rate=0.0005)
 
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    history = model.fit(x_train, y_train, batch_size=32, epochs=10, shuffle=True)

    model.evaluate(x_test, y_test)

    # Above is actually worse, adding Batch Normalisation / Dropout suggested
    input_layer = layers.Input(shape=(32, 32, 3))   
    x = layers.Conv2D(
            filters=32,
            kernel_size=3,
            strides=1,
            padding="same"
            )(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(
            filters=32,
            kernel_size=3,
            strides=2,
            padding="same"
            )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding="same"
            )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2D(
            filters=64,
            kernel_size=3,
            strides=2,
            padding="same"
            )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(rate=0.5)(x)
    output_layer=layers.Dense(10, activation="softmax")(x)
    model = models.Model(input_layer, output_layer)

    opt = optimizers.Adam(learning_rate=0.0005)
 
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    history = model.fit(x_train, y_train, batch_size=32, epochs=10, shuffle=True)

    model.evaluate(x_test, y_test)





