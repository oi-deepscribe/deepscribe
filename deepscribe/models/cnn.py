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


class CNNAugment(ParameterModel, ABC):
    """

    Subclass of ParameterModel that trains a CNN image classification network with a data augmentation routine.

    """

    def _train_model(
        self,
        x_train: np.array,
        y_train: np.array,
        x_val: np.array,
        y_val: np.array,
        model: kr.Model,
        params: Dict,
    ) -> kr.callbacks.History:

        if "seed" in params:
            tf.random.set_seed(params["seed"])

        callbacks = (
            [
                kr.callbacks.EarlyStopping(
                    monitor="val_loss", patience=params["early_stopping"]
                )
            ]
            if "early_stopping" in params
            else []
        )

        # adding

        if "reduce_lr" in params:
            callbacks.append(
                kr.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", min_delta=1e-4, patience=params["reduce_lr"]
                )
            )

        # logging params to wandb - not syncing, active syncing causes
        # slurm to not terminate the job
        os.environ["WANDB_MODE"] = "dryrun"

        wandb.init(project="deepscribe", config=params)

        callbacks.append(WandbCallback())

        if "reweight" in params:
            class_weights_arr = compute_class_weight(
                "balanced", np.unique(y_train), y_train
            )
            class_weight_dict = dict(enumerate(class_weights_arr))
        else:
            class_weight_dict = None

        # use image data generator to perform random translations

        shear_range = params.get("shear", 0.0)
        zoom_range = params.get("zoom", 0.0)
        width_shift = params.get("width-shift", 0.0)
        height_shift = params.get("width-shift", 0.0)
        data_gen = kr.preprocessing.image.ImageDataGenerator(
            shear_range=shear_range,
            zoom_range=zoom_range,
            width_shift_range=width_shift,
            height_shift_range=height_shift,
        )

        history = model.fit_generator(
            data_gen.flow(x_train, y=y_train),
            steps_per_epoch=x_train.shape[0] / params["batch_size"],
            epochs=params["epochs"],
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            class_weight=class_weight_dict,
        )

        return history


class CNN2Conv(CNNAugment):
    """

    Subclass of CNNAugment that implements a 2-layer CNN model.

    """

    def _build_model(self, params: Dict, img_shape: tuple = None) -> kr.Model:
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
        model.add(
            kr.layers.Dense(params["dense_size"], activation=params["activation"])
        )
        # model.add(kr.layers.Dense(512, activation='relu'))
        model.add(kr.layers.Dense(params["num_classes"], activation="softmax"))

        # TODO: set learning rat

        return model


class VGG16(CNNAugment):
    """

    Subclass of CNNAugment that transfers learned weights from VGG16.

    """

    def _build_model(self, params: Dict, img_shape: tuple = None) -> kr.Model:

        base_model = kr.applications.vgg16.VGG16(
            weights="imagenet" if params.get("transfer", False) else None,
            include_top=False,
            input_shape=img_shape,
        )

        x = base_model.output

        # TODO: freeze dynamic number of layers based on config file

        x = kr.layers.GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer

        x = kr.layers.Dropout(params["dropout"])(x)

        x = kr.layers.Dense(params["dense_size"], activation=params["activation"])(x)
        predictions = kr.layers.Dense(params["num_classes"], activation="softmax")(x)

        # freeze layers
        if params.get("transfer", False):
            for layer in base_model.layers:
                layer.trainable = False

        # TODO: set learning rate

        model = kr.Model(inputs=base_model.input, outputs=predictions)

        return model


class VGG19(CNNAugment):
    """

        Subclass of CNNAugment that transfers learned weights from VGG19.

        """

    def _build_model(self, params: Dict, img_shape: tuple = None) -> kr.Model:

        base_model = kr.applications.vgg19.VGG19(
            weights="imagenet" if params.get("transfer", False) else None,
            include_top=False,
            input_shape=img_shape,
        )

        x = base_model.output

        # TODO: freeze dynamic number of layers based on config file

        x = kr.layers.GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer

        x = kr.layers.Dropout(params["dropout"])(x)

        for i in range(params["n_dense"]):
            x = kr.layers.Dense(params["dense_size"], activation=params["activation"])(
                x
            )

        predictions = kr.layers.Dense(params["num_classes"], activation="softmax")(x)

        # freeze layers if transferring fixed weights
        if params.get("transfer", False):
            for layer in base_model.layers:
                layer.trainable = False

        # TODO: set learning rate

        model = kr.Model(inputs=base_model.input, outputs=predictions)

        return model


class ResNet50(CNNAugment):
    """
    Subclass of CNNAugment using architecture from ResNet50

    """

    def _build_model(self, params: Dict, img_shape: tuple = None) -> kr.Model:

        base_model = kr.applications.resnet50.ResNet50(
            weights="imagenet" if params.get("transfer", False) else None,
            include_top=False,
            input_shape=img_shape,
        )

        x = base_model.output

        # TODO: freeze dynamic number of layers based on config file

        x = kr.layers.GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer

        x = kr.layers.Dropout(params["dropout"])(x)

        for i in range(params["n_dense"]):
            x = kr.layers.Dense(params["dense_size"], activation=params["activation"])(
                x
            )

        predictions = kr.layers.Dense(params["num_classes"], activation="softmax")(x)

        # freeze layers if transferring fixed weights
        if params.get("transfer", False):
            for layer in base_model.layers:
                layer.trainable = False

        # TODO: set learning rate

        model = kr.Model(inputs=base_model.input, outputs=predictions)

        return model


class ResNet50V2(CNNAugment):
    """
    Subclass of CNNAugment using architecture from ResNet50

    """

    def _build_model(self, params: Dict, img_shape: tuple = None) -> kr.Model:

        base_model = kr.applications.resnet_v2.ResNet50V2(
            weights="imagenet" if params.get("transfer", False) else None,
            include_top=False,
            input_shape=img_shape,
        )

        x = base_model.output

        # TODO: freeze dynamic number of layers based on config file

        x = kr.layers.GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer

        x = kr.layers.Dropout(params["dropout"])(x)

        for i in range(params["n_dense"]):
            x = kr.layers.Dense(params["dense_size"], activation=params["activation"])(
                x
            )

        predictions = kr.layers.Dense(params["num_classes"], activation="softmax")(x)

        # freeze layers if transferring fixed weights
        if params.get("transfer", False):
            for layer in base_model.layers:
                layer.trainable = False

        # TODO: set learning rate

        model = kr.Model(inputs=base_model.input, outputs=predictions)

        return model


class ResNet18(CNNAugment):
    """
    Shallower ResNet architecture.

    """

    def _build_model(self, params: Dict, img_shape: tuple = None) -> kr.Model:

        img_input = kr.layers.Input(shape=img_shape)
        x = layers.ZeroPadding2D(padding=(3, 3), name="conv1_pad")(img_input)
        x = layers.Conv2D(
            64,
            (7, 7),
            strides=(2, 2),
            padding="valid",
            kernel_initializer="he_normal",
            name="conv1",
        )(x)
        x = layers.BatchNormalization(name="bn_conv1")(x)
        x = layers.Activation("relu")(x)
        x = layers.ZeroPadding2D(padding=(1, 1), name="pool1_pad")(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
        # TODO: read these from params
        # default values from the original ResNet50 implementation.
        x = conv_block(x, 3, [64, 64, 256], stage=2, block="a", strides=(1, 1))
        x = identity_block(x, 3, [64, 64, 256], stage=2, block="b")
        x = identity_block(x, 3, [64, 64, 256], stage=2, block="c")

        x = conv_block(x, 3, [128, 128, 512], stage=3, block="a")
        x = identity_block(x, 3, [128, 128, 512], stage=3, block="b")
        x = identity_block(x, 3, [128, 128, 512], stage=3, block="c")

        x = kr.layers.GlobalAveragePooling2D()(x)

        x = kr.layers.Dropout(params["dropout"])(x)

        for i in range(params["n_dense"]):
            x = kr.layers.Dense(params["dense_size"], activation=params["activation"])(
                x
            )

        predictions = kr.layers.Dense(params["num_classes"], activation="softmax")(x)

        model = kr.Model(inputs=img_input, outputs=predictions)

        return model
