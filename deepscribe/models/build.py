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


def model_from_params(params: Dict, img_shape: Tuple = None) -> kr.Model:
    if params["architecture"] == "resnet18":
        model = build_resnet18(params, img_shape=img_shape)
    else:
        raise ValueError(
            f"architecture {params['architecture']} is not a valid option."
        )

    # compiling model based on parameter dict

    optimizer_type = params.get("optimizer", "adam")

    if optimizer_type == "adam":
        optimizer = kr.optimizers.Adam(lr=params.get("lr", 0.001))
    else:
        raise ValueError(f"optimizer {optimizer_type} is not a valid option.")

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=[
            "acc",
            kr.metrics.SparseTopKCategoricalAccuracy(k=3, name="top3_acc"),
            kr.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_acc"),
        ],
    )

    return model


def build_resnet18(params: Dict, img_shape: tuple = None) -> kr.Model:

    # set up regularizer

    reg_l1 = params.get("l1", 0.0)  # default param - no penalty
    reg_l2 = params.get("l1", 0.0)  # default param - no penalty

    regularizer = kr.regularizers.l1_l2(l1=reg_l1, l2=reg_l2)

    img_input = kr.layers.Input(shape=img_shape)
    x = layers.ZeroPadding2D(padding=(3, 3), name="conv1_pad")(img_input)
    x = layers.Conv2D(
        64,
        (7, 7),
        strides=(2, 2),
        padding="valid",
        kernel_initializer="he_normal",
        name="conv1",
        kernel_regularizer=regularizer,
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
