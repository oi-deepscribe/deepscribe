# implementing model-building functions.
import tensorflow.keras as kr
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
from typing import Dict, Tuple
import wandb
from wandb.keras import WandbCallback
import os
from sklearn.utils.class_weight import compute_class_weight
from abc import ABC
from .parametermodel import ParameterModel
from .blocks import conv_block, identity_block
from imblearn.over_sampling import RandomOverSampler
from focal_loss import SparseCategoricalFocalLoss


def model_from_params(params: Dict, img_shape: Tuple = None) -> kr.Model:
    if params["architecture"] == "resnet18":
        model = build_resnet18(params, img_shape=img_shape)
    elif params["architecture"] == "cnn2conv":
        model = build_cnn2conv(params, img_shape=img_shape)
    elif params["architecture"] == "resnet50":
        model = build_resnet50(params, img_shape=img_shape)
    else:
        raise ValueError(
            f"architecture {params['architecture']} is not a valid option."
        )

    # compiling model based on parameter dict

    optimizer_type = params.get("optimizer", "adam")

    if optimizer_type == "adam":
        optimizer = kr.optimizers.Adam(lr=params.get("lr", 0.001))
    elif optimizer_type == "amsgrad":
        optimizer = kr.optimizers.Adam(lr=params.get("lr", 0.001), amsgrad=True)
    elif optimizer_type == "rmsprop":
        optimizer = kr.optimizers.RMSprop(lr=params.get("lr", 0.001))
    elif optimizer_type == "adamax":
        optimizer = kr.optimizers.Adamax(lr=params.get("lr", 0.001))
    elif optimizer_type == "sgd":
        optimizer = kr.optimizers.SGD(
            lr=params.get("lr", 0.001), momentum=params.get("momentum", 0.0)
        )
    else:
        raise ValueError(f"optimizer {optimizer_type} is not a valid option.")

    # using regular sparse categorical crossentropy
    # even though these should be equivalent.

    # also a good candidate for a walrus operator.

    focal_gamma = params.get("focal", 0.0)

    loss = (
        SparseCategoricalFocalLoss(focal_gamma)
        if focal_gamma > 0.0
        else "sparse_categorical_crossentropy"
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            "acc",
            kr.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc"),
            kr.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_acc"),
        ],
    )

    return model


def build_resnet50(params: Dict, img_shape: tuple = None) -> kr.Model:

    base_model = kr.applications.resnet_v2.ResNet50V2(
        weights=None, include_top=False, input_shape=img_shape,
    )

    x = base_model.output
    x = kr.layers.GlobalAveragePooling2D()(x)
    predictions = kr.layers.Dense(params["num_classes"], activation="softmax")(x)

    model = kr.Model(inputs=base_model.input, outputs=predictions)
    return model


def build_cnn2conv(params: Dict, img_shape: tuple = None) -> kr.Model:

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

    return model


def build_resnet18(params: Dict, img_shape: tuple = None) -> kr.Model:

    # set up regularizer

    reg_l1 = params.get("l1", 0.0)  # default param - no penalty
    reg_l2 = params.get("l1", 0.0)  # default param - no penalty

    regularizer = kr.regularizers.l1_l2(l1=reg_l1, l2=reg_l2)

    img_input = kr.layers.Input(shape=img_shape)
    input_dropout = kr.layers.Dropout(params.get("input_dropout", 0.0))(img_input)
    x = layers.ZeroPadding2D(padding=(3, 3), name="conv1_pad")(input_dropout)
    x = layers.Conv2D(
        64,
        (7, 7),
        strides=(2, 2),
        padding="valid",
        kernel_initializer="he_normal",
        name="conv1",
        kernel_regularizer=regularizer,
        bias_regularizer=regularizer,
        activity_regularizer=regularizer,
    )(x)
    x = layers.BatchNormalization(name="bn_conv1")(x)
    x = layers.Activation("relu")(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name="pool1_pad")(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    # TODO: read these from params
    # default values from the original ResNet50 implementation.
    x = conv_block(
        x,
        3,
        [64, 64, 256],
        stage=2,
        block="a",
        strides=(1, 1),
        regularizer=regularizer,
    )
    x = identity_block(x, 3, [64, 64, 256], stage=2, block="b", regularizer=regularizer)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block="c", regularizer=regularizer)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block="a", regularizer=regularizer)
    x = identity_block(
        x, 3, [128, 128, 512], stage=3, block="b", regularizer=regularizer
    )
    x = identity_block(
        x, 3, [128, 128, 512], stage=3, block="c", regularizer=regularizer
    )

    x = kr.layers.GlobalAveragePooling2D()(x)

    x = kr.layers.Dropout(params.get("dropout", 0.0))(x)

    for _ in range(params.get("n_dense", 0)):
        x = kr.layers.Dense(
            params.get("dense_size", 512), activation=params["activation"]
        )(x)

    predictions = kr.layers.Dense(params["num_classes"], activation="softmax")(x)

    model = kr.Model(inputs=img_input, outputs=predictions)

    return model
