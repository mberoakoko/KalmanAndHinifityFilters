import functools

import numpy as np
from numpy.typing import NDArray
from typing import NamedTuple, Callable, LiteralString


class PredictionState(NamedTuple):
    x: NDArray
    p: NDArray

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

def update(H: NDArray, R: NDArray) -> Callable[[PredictionState, NDArray], PredictionState]:
    def _update(state: PredictionState, z: NDArray) -> PredictionState:
        x, P = state
        y = z - H @ x  # Innovation
        S = H @ P @ H.T + R  # Innovation covariance
        K = P @ H.T @ np.linalg.inv(S)  # Kalman Gain

        new_x = x + K @ y
        new_P = (np.eye(len(x)) - K @ H) @ P
        return PredictionState(new_x, new_P)

    return _update

def kalman_step(predict_func: PredictProto, update_func: UpdateProto) -> Callable[[PredictionState, NDArray], PredictionState]:
    def inner_step(state: PredictionState, z: NDArray):
        return update_func(predict_func(state), z)
    return inner_step

def example_execution():
    F = np.array([[1, 1], [0, 1]])  # State transition
    H = np.array([[1, 0]])  # Observation model
    Q = np.eye(2) * 0.01  # Process noise
    R = np.array([[0.1]])  # Measurement noise

    step = kalman_step(predict_func=predict(F, Q), update_func=update(H, R))
    initial_belief = PredictionState(np.array([0, 0]), np.eye(2))
    measurements = [np.array([1]), np.array([2]), np.array([3])]

    def run_filter(beliefs: list[PredictionState], measurement: NDArray):
        last_belief = beliefs[-1]
        return beliefs + [step(last_belief, measurement)]

    final_history = functools.reduce(run_filter, measurements, [initial_belief])
    print(final_history)

example_execution()


