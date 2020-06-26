import tensorflow.keras as kr
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
from typing import Dict, Tuple
import cv2
import wandb
from wandb.keras import WandbCallback
import os
from sklearn.utils.class_weight import compute_class_weight
from abc import ABC
from .build import model_from_params
from .augment import add_random_shadow
from imblearn.over_sampling import RandomOverSampler

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def build_train_params(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict,
) -> Tuple[kr.callbacks.History, kr.Model]:

    model = model_from_params(params, img_shape=x_train.shape[1:])

    history = train_from_params(x_train, y_train, x_val, y_val, model, params)

    return history, model


def train_from_params(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    model: kr.Model,
    params: Dict,
) -> kr.callbacks.History:

    if "seed" in params:
        tf.random.set_seed(params["seed"])

    callbacks = [kr.callbacks.TerminateOnNaN()]

    if params.get("early_stopping", 0) > 0:
        callbacks.append(
            kr.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=params["early_stopping"],
                restore_best_weights=True,
            )
        )

    # adding

    if params.get("reduce_lr", 0) > 0:
        callbacks.append(
            kr.callbacks.ReduceLROnPlateau(
                monitor="val_loss", min_delta=1e-4, patience=params["reduce_lr"]
            )
        )

    # logging params to wandb - not syncing, active syncing causes
    # slurm to not terminate the job
    # disabled now that we're not using slurm
    # os.environ["WANDB_MODE"] = "dryrun"

    wandb.init(project="deepscribe", config=params)

    callbacks.append(WandbCallback())

    if params.get("reweight", False):
        class_weights_arr = compute_class_weight(
            "balanced", np.unique(y_train), y_train
        )
        class_weight_dict = dict(enumerate(class_weights_arr))
    else:
        class_weight_dict = None

    # use image data generator to perform random translations

    shear_range = params.get("shear", 0.0)
    zoom_range = params.get("zoom", 0.0)
    width_shift = params.get("width_shift", 0.0)
    height_shift = params.get("width_shift", 0.0)
    rotation = params.get("rotation_range", 0.0)
    shadow_val_range = params.get("shadow_val_range", [0.0, 0.0])
    num_shadows = params.get("num_shadows", 0.0)

    data_gen = kr.preprocessing.image.ImageDataGenerator(
        shear_range=shear_range,
        zoom_range=zoom_range,
        width_shift_range=width_shift,
        height_shift_range=height_shift,
        rotation_range=rotation,
        fill_mode="constant",
        preprocessing_function=lambda img: add_random_shadow(
            img, shadow_val_range, num_shadows=num_shadows
        ),
        cval=0.0,
    )

    # oversample training data

    if params.get("oversample", False):
        # need to flatten to use with estimator
        # this is probably very memory inefficient
        x_train_flat, y_train = RandomOverSampler(random_state=0).fit_resample(
            x_train.reshape((x_train.shape[0], -1)), y_train
        )

        # new amount of data!
        x_train = x_train_flat.reshape(
            tuple(y_train.shape[:1]) + tuple(x_train.shape[1:])
        )

    # print(f"should be {x_train.shape[0] / params["bsize"]} steps per epoch, {x_train.shape[0]} data pts")

    history = model.fit(
        data_gen.flow(x_train, y=y_train, batch_size=params.get("bsize", 32)),
        steps_per_epoch=x_train.shape[0] / params.get("bsize", 32),
        epochs=params["epochs"],
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        class_weight=class_weight_dict,
    )

    return history
