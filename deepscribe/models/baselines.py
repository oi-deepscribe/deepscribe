# baseline models for single-image classification.

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
        model.add(kr.layers.Dense(layer_size, activation="relu"))

    # final layer
    model.add(kr.layers.Dense(num_classes, activation="softmax"))

    return model


def cnn_classifier(x_train, y_train, x_val, y_val, params):
    """Builds CNN classifier according to parameter dictionary.

    Parameters
    ----------
    input_shape : tuple
        Input image shape
    num_classes : int
        Number of output classes
    params : type
        Description of parameter `params`.

    Returns
    -------
    type
        Description of returned object.

    """

    model = kr.models.Sequential()
    model.add(
        kr.layers.Conv2D(
            params["conv1_kernels"],
            kernel_size=(params["conv1_ksize"], params["conv1_ksize"]),
            strides=(params["conv1_stride"], params["conv1_stride"]),
            activation=params["activation"],
        )
    )
    model.add(
        kr.layers.MaxPooling2D(
            pool_size=(params["pool1_size"], params["pool1_size"]),
            strides=(params["pool1_stride"], params["pool1_stride"]),
        )
    )
    model.add(kr.layers.BatchNormalization())
    model.add(kr.layers.Dropout(params["dropout"]))
    model.add(
        kr.layers.Conv2D(
            params["conv2_kernels"],
            kernel_size=(params["conv2_ksize"], params["conv2_ksize"]),
            strides=(params["conv2_stride"], params["conv2_stride"]),
            activation=params["activation"],
        )
    )
    model.add(kr.layers.BatchNormalization())
    model.add(
        kr.layers.MaxPooling2D(
            pool_size=(params["pool2_size"], params["pool2_size"]),
            strides=(params["pool2_stride"], params["pool2_stride"]),
        )
    )

    model.add(kr.layers.Dropout(params["dropout"]))
    model.add(kr.layers.Flatten())
    model.add(kr.layers.Dense(params["dense_size"], activation=params["activation"]))
    # model.add(kr.layers.Dense(512, activation='relu'))
    model.add(kr.layers.Dense(params["num_classes"], activation="softmax"))

    # TODO: set learning rate
    model.compile(
        optimizer=params["optimizer"],
        loss="categorical_crossentropy",
        metrics=["acc", kr.metrics.AUC()],
    )

    # TODO: early stopping
    history = model.fit(
        x_train,
        y_train,
        batch_size=params["batch_size"],
        epochs=params["epochs"],
        validation_data=(x_val, y_val),
    )

    return model
