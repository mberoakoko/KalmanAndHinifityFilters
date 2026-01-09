import functools
import numpy as np
from numpy.typing import NDArray
from enum import Enum, auto
from typing import Callable, NamedTuple
from src.utils import iterate as abstract_iter

class PredictionState(NamedTuple):
    x: NDArray
    u: NDArray
    p: NDArray

class CovUpdateLaw(Enum):
    LAW_1: auto
    LAW_2: auto
    LAw_3: auto



type PredictProto = Callable[[PredictionState], PredictionState]
type UpdateProto = Callable[[PredictionState, NDArray], PredictionState]

def predict(F: NDArray, G: NDArray, Q: NDArray) -> Callable[[PredictionState], PredictionState]:
    def _inner_pred_func(state: PredictionState):
        x, u, p = state.x, state.u,  state.p
        return PredictionState(
            x=F @ x + G @ u,
            p=F @ p @ F.T + Q
        )
    return _inner_pred_func


def __covariance_update(H: NDArray, K: NDArray, P: NDArray, R: NDArray, cov_update_law: CovUpdateLaw = CovUpdateLaw.LAW_1) -> NDArray:
    match cov_update_law:
        case CovUpdateLaw.LAW_1:
            n = K.shape[0]
            return (np.eye(n) - K @ H) @ P @ np.linalg.inv(np.eye(n) - K @ H).T + K @ R @ K
        case CovUpdateLaw.LAW_2:
            return np.linalg.inv(np.linalg.inv(P) + H.T @ R @ H)

        case CovUpdateLaw.LAw_3:
            n = K.shape[0]
            return (np.eye(n) - K @ H) @ P

def update(H: NDArray, R: NDArray, cov_update_law: CovUpdateLaw = CovUpdateLaw.LAW_1) -> Callable[[PredictionState, NDArray], PredictionState]:
    def _update_law(state: PredictionState, z: NDArray) -> PredictionState:
        x, _,  P = state
        y = z - H @ x  # Innovation
        S = H @ P @ H.T + R  # Innovation covariance
        K = P @ H.T @ np.linalg.inv(S)  # Kalman Gain

        new_x = x + K @ y
        new_P = __covariance_update(H, K, P, R, cov_update_law)
        return PredictionState(new_x, new_P)

    return _update_law

def __kalman_step(predict_function: PredictProto, update_function: UpdateProto):
    def inner_func(state: PredictionState, z: NDArray) -> PredictionState:
        return update_function(predict_function(state), z)
    return inner_func

type FilterStep = Callable[[PredictionState, NDArray], PredictionState]

def create_kalman_step(
        F: NDArray,
        G: NDArray,
        Q: NDArray,
        H: NDArray,
        R: NDArray,
        covariance_update_law: CovUpdateLaw
) -> FilterStep:
    return __kalman_step(
        predict_function=predict(F, G, Q),
        update_function=update(H, R, covariance_update_law)
    )

def filter_measurements(filter_step: FilterStep, initial_belief: PredictionState,  measurements: NDArray) -> list[PredictionState]:
    def __run_filter(beliefs: list[PredictionState], measurement: NDArray) -> list[PredictionState]:
        last_belief = beliefs[-1]
        return beliefs + [filter_step(last_belief, measurement)]
    return functools.reduce(__run_filter, measurements, [initial_belief])

