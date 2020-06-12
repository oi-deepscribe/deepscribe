# locations of tf.keras implementations of ResNet blocks.

# from https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet50.py

from tensorflow.keras import layers
import tensorflow as tf
from typing import List, Tuple


def identity_block(
    input_tensor: tf.Tensor,
    kernel_size: int,
    filters: List[int],
    stage: int,
    block: str,
    regularizer=None,
):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    x = layers.Conv2D(
        filters1,
        (1, 1),
        kernel_initializer="he_normal",
        name=conv_name_base + "2a",
        kernel_regularizer=regularizer,
        bias_regularizer=regularizer,
        activity_regularizer=regularizer,
    )(input_tensor)
    x = layers.BatchNormalization(name=bn_name_base + "2a")(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(
        filters2,
        kernel_size,
        padding="same",
        kernel_initializer="he_normal",
        name=conv_name_base + "2b",
        kernel_regularizer=regularizer,
        bias_regularizer=regularizer,
        activity_regularizer=regularizer,
    )(x)
    x = layers.BatchNormalization(name=bn_name_base + "2b")(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(
        filters3,
        (1, 1),
        kernel_initializer="he_normal",
        name=conv_name_base + "2c",
        kernel_regularizer=regularizer,
        bias_regularizer=regularizer,
        activity_regularizer=regularizer,
    )(x)
    x = layers.BatchNormalization(name=bn_name_base + "2c")(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation("relu")(x)
    return x


def conv_block(
    input_tensor: tf.Tensor,
    kernel_size: int,
    filters: List[int],
    stage: int,
    block: str,
    strides: Tuple = (2, 2),
    regularizer=None,
):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters

    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    x = layers.Conv2D(
        filters1,
        (1, 1),
        strides=strides,
        kernel_initializer="he_normal",
        name=conv_name_base + "2a",
        kernel_regularizer=regularizer,
        bias_regularizer=regularizer,
        activity_regularizer=regularizer,
    )(input_tensor)
    x = layers.BatchNormalization(name=bn_name_base + "2a")(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(
        filters2,
        kernel_size,
        padding="same",
        kernel_initializer="he_normal",
        name=conv_name_base + "2b",
        kernel_regularizer=regularizer,
        bias_regularizer=regularizer,
        activity_regularizer=regularizer,
    )(x)
    x = layers.BatchNormalization(name=bn_name_base + "2b")(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(
        filters3,
        (1, 1),
        kernel_initializer="he_normal",
        name=conv_name_base + "2c",
        kernel_regularizer=regularizer,
        bias_regularizer=regularizer,
        activity_regularizer=regularizer,
    )(x)
    x = layers.BatchNormalization(name=bn_name_base + "2c")(x)

    shortcut = layers.Conv2D(
        filters3,
        (1, 1),
        strides=strides,
        kernel_initializer="he_normal",
        name=conv_name_base + "1",
        kernel_regularizer=regularizer,
        bias_regularizer=regularizer,
        activity_regularizer=regularizer,
    )(input_tensor)
    shortcut = layers.BatchNormalization(name=bn_name_base + "1")(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation("relu")(x)

    return x
