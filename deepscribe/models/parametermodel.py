from abc import ABC
from typing import Dict, Tuple
import numpy as np
import tensorflow.keras as kr


# generic class for a model that is initialized from a parameter dictionary.
# allows modularity and code reuse when building and training the model.
class ParameterModel(ABC):
    def __init__(self):
        pass

    def _build_model(self, params: Dict) -> kr.Model:
        raise NotImplementedError

    # avoiding establishing an internal model "state" here in the object.
    def _train_model(
        self,
        x_train: np.array,
        y_train: np.array,
        x_val: np.array,
        y_val: np.array,
        model: kr.Model,
        params: Dict,
    ) -> kr.callbacks.History:
        raise NotImplementedError

    def __call__(
        self,
        x_train: np.array,
        y_train: np.array,
        x_val: np.array,
        y_val: np.array,
        params: Dict,
    ) -> Tuple[kr.callbacks.History, kr.models.Model]:
        model = self._build_model(params)

        history = self._train_model(x_train, y_train, x_val, y_val, model, params)

        return history, model
