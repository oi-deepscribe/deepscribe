import tensorflow.keras as kr
import tensorflow as tf
import numpy as np
from typing import Dict, Tuple
import wandb
from wandb.keras import WandbCallback
import os
from sklearn.utils.class_weight import compute_class_weight
from abc import ABC
from .parametermodel import ParameterModel


class CNNAugment(ParameterModel, ABC):
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
        # logging params to wandb - not syncing, active syncing causes
        # slurm to not terminate the job
        os.environ["WANDB_MODE"] = "dryrun"
        # adding more stuff to the params dict
        params["input_shape"] = x_train.shape
        params["SLURM_RUN"] = os.environ.get("SLURM_JOB_ID", "NONE")

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
        data_gen = kr.preprocessing.image.ImageDataGenerator(
            shear_range=shear_range, zoom_range=zoom_range
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
    def _build_model(self, params: Dict) -> kr.Model:
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

        # TODO: set learning rate
        model.compile(
            optimizer=params["optimizer"],
            loss="sparse_categorical_crossentropy",
            metrics=["acc", kr.metrics.TopKCategoricalAccuracy(k=params.get("k", 3))],
        )

        return model