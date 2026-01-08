import functools
import numpy as np
from numpy.typing import NDArray
from typing import Callable, NamedTuple
from src.filter_impl.generic_kalman_filter import PredictionState


type PredictProto = Callable[[PredictionState], PredictionState]
type UpdateProto = Callable[[PredictionState, NDArray], PredictionState]

def predict(F: NDArray, Q: NDArray) -> Callable[[PredictionState], PredictionState]:
    def _inner_pred_func(state: PredictionState):
        x, p = state.x, state.p
        return PredictionState(
            x=F @ x,
            p=F @ p @ F.T + Q
        )
    return _inner_pred_func
