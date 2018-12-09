#baseline models for single-image classification.

import tensorflow as tf
import tensorflow.keras as kr

def mlp_classifier(input_shape, hidden_layers, layer_size, num_classes):
    """Multi-layer perceptron model. Flattens images first.

    Parameters
    ----------
    input_shape : tuple of integers
        Input shape of model
    hidden_layers : int
        Number of hidden layers.
    layer_size : int
        Hidden layer size.
    num_classes : int
        number of output classes.

    Returns
    -------
    kr.models.Sequential
        Initialized model.

    """

    model = kr.models.Sequential()
    # flatten the input image
    model.add(kr.layers.Flatten(input_shape=input_shape))

    for _ in range(hidden_layers):
        model.add(kr.layers.Dense(layer_size, activation='relu'))

    # final layer
    model.add(kr.layers.Dense(num_classes, activation='softmax'))

    return model

def cnn_classifier(input_shape, num_classes):
    """Compiles two-layer CNN classifier with the provided input shape and number of classes.

    Parameters
    ----------
    input_shape : tuple of integers
        Input shape of model.
    num_classes : int
        Number of output classes.

    Returns
    -------
    kr.models.Sequential
        Initialized model.

    """



    model = kr.models.Sequential()
    model.add(kr.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu',
                     input_shape=input_shape))
    model.add(kr.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(kr.layers.Conv2D(64, (5, 5), activation='relu'))
    model.add(kr.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(kr.layers.Flatten())
    model.add(kr.layers.Dense(512, activation='relu'))
    model.add(kr.layers.Dense(num_classes, activation='softmax'))

    return model
