from abc import ABC
from typing import Dict, Tuple
import numpy as np
import tensorflow.keras as kr


class ParameterModel(ABC):
    """
    generic class for a model that is initialized from a parameter dictionary.
    allows modularity and code reuse when building and training the model.

    As of this writing, only implements the model building and training function - no model persistance is actually
    taken into account.

    Must implement the functions _build_model and _train_model. 

    """

    def __init__(self):
        pass

    def _build_model(self, params: Dict) -> kr.Model:
        """

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
        model = self._build_model(params)

        history = self._train_model(x_train, y_train, x_val, y_val, model, params)

        return history, model
