from abc import ABC
from typing import Dict, Tuple
import numpy as np
import tensorflow.keras as kr

# DEPRECATED. I was trying out some callable inheritance stuff, but it turned out to be pretty overkill.


class ParameterModel(ABC):
    """
    generic class for a model that is initialized from a parameter dictionary.
    allows modularity and code reuse when building and training the model.

    As of this writing, only implements the model building and training function - no model persistence is actually
    taken into account.

    Must implement the functions _build_model and _train_model. 

    """

    def __init__(self):
        pass

    def _build_model(self, params: Dict, img_shape: tuple = None) -> kr.Model:
        """

        :param img_shape: tuple containing image shape
        :param params: Dictionary containing model parameter values.

        returns compiled Keras model.

        """
        raise NotImplementedError

    # avoiding establishing an internal model "state" here in the object.
    def _train_model(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        model: kr.Model,
        params: Dict,
    ) -> kr.callbacks.History:
        """

        :param x_train: training image data, with shape [n_images, x_dim, y_dim, n_channels]
        :param y_train: categorical variables with shape [n_images,]
        :param x_val: validation image data, with shape [n_images, x_dim, y_dim, n_channels]
        :param y_val: categorical variables with shape [n_images,]
        :param model: tf.Keras model.
        :param params: parameter dictionary.
        """
        raise NotImplementedError

    def __call__(
        self,
        x_train: np.array,
        y_train: np.array,
        x_val: np.array,
        y_val: np.array,
        params: Dict,
    ) -> Tuple[kr.callbacks.History, kr.models.Model]:
        """

        :param x_train: training image data, with shape [n_images, x_dim, y_dim, n_channels]
        :param y_train: categorical variables with shape [n_images,]
        :param x_val: validation image data, with shape [n_images, x_dim, y_dim, n_channels]
        :param y_val: categorical variables with shape [n_images,]
        :param model: tf.Keras model.
        :param params: parameter dictionary.
        """
        model = self._build_model(params, img_shape=x_train.shape[1:])

        # compile model here!

        # TODO: pick optimizer type

        optimizer_type = params.get("optimizer", "adam")

        if optimizer_type == "adam":

            optimizer = kr.optimizers.Adam(lr=params.get("lr", 0.001))

        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=[
                "acc",
                kr.metrics.SparseTopKCategoricalAccuracy(k=params.get("k", 3)),
            ],
        )

        history = self._train_model(x_train, y_train, x_val, y_val, model, params)

        return history, model
