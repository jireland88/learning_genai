import numpy as np
from tensorflow.keras import datasets, utils, layers, models

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
    # x_train has shape (50000, 32, 32, 3) = (number of training examples, height, width, color channels)
    # Flatten, "flattens" the input from a (32, 32, 3) tensor to a 32x32x3 length vector
    # Dense requires a vector input. Every unit in the dense layer is connected to every unit in the previous layer
    # Each connection carries a weight.
    # Output of each unit in the Dense layer is the activation function applied to the weighted sum of the inputs

    # TODO: Notes on activation functions
    

    input_layer = layers.Input(shape=(32, 32, 3))
    x = layers.Flatten()(input_layer)
    x = layers.Dense(units=200, activation="relu")(x)
    x = layers.Dense(units=150, activation="relu")(x)
    output_layer = layers.Dense(units=10, activation="softmax")(x)
    
    model = models.Model(input_layer, output_layer)



