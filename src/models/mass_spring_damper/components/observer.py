import control
import numpy as np
from numpy.typing import NDArray

def _generate_optimal_observer_gain(system: control.StateSpace, Q: NDArray, R: NDArray) -> NDArray:
    if np.linalg.matrix_rank(control.obsv(system.A, system.C)) < system.A.shape[0]:
        raise ValueError("The system is not observable, controllable matrix is rank deficient")
    P, _, _ = control.lqe(system.A, system.C, Q, R)
    return P

