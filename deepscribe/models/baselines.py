# baseline models for single-image classification.

import tensorflow.keras as kr
import numpy as np
from typing import Dict, Tuple
import wandb
from wandb.keras import WandbCallback
import os


def cnn_classifier_2conv(
    x_train: np.array, y_train: np.array, x_val: np.array, y_val: np.array, params: Dict
) -> Tuple[kr.callbacks.History, kr.models.Model]:
    """


    :param x_train:
    :param y_train:
    :param x_val:
    :param y_val:
    :param params:
    :param labels:
    :return:
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
        loss="sparse_categorical_crossentropy",
        metrics=["acc", kr.metrics.TopKCategoricalAccuracy(k=3)],
    )

    callbacks = (
        [
            kr.callbacks.EarlyStopping(
                monitor="val_loss", patience=params["early_stopping"]
            )
        ]
        if "early_stopping" in params
        else []
    )
    # logging params to wandb - not syncing
    os.environ["WANDB_MODE"] = "dryrun"
    wandb.init(project="deepscribe", config=params)

    callbacks.append(WandbCallback())

    history = model.fit(
        x_train,
        y_train,
        batch_size=params["batch_size"],
        epochs=params["epochs"],
        validation_data=(x_val, y_val),
        callbacks=callbacks,
    )

    return history, model
