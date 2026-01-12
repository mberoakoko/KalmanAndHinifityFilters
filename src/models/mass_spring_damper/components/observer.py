import control
import numpy as np
from numpy.typing import NDArray
import dataclasses


def _generate_optimal_observer_gain(system: control.StateSpace, Q: NDArray, R: NDArray) -> NDArray:
    if np.linalg.matrix_rank(control.obsv(system.A, system.C)) < system.A.shape[0]:
        raise ValueError("The system is not observable, controllable matrix is rank deficient")
    P, _, _ = control.lqe(system.A, system.C, Q, R)
    return P

@dataclasses.dataclass
class LuenbergerObserver:
    system: control.StateSpace
    Q: NDArray
    R: NDArray
    L: NDArray | None = dataclasses.field(init=False)

    def __post_init__(self):
        self.L = _generate_optimal_observer_gain(self.system, self.Q, self.R)

    def __update(self, t: float, x: NDArray, u: NDArray , params: dict = None) -> NDArray:
        u_, y = u
        return self.system.A @ x + self.system.B @ u_ @ (y - self.L @ x)

    def __output(self, t: float, x: NDArray, u: NDArray, params:dict = None) -> NDArray:
        return x


    def create_observer(self) -> control.NonlinearIOSystem:
        return control.NonlinearIOSystem(
            self.__update, self.__output,
            name="Observer",
            inputs=("y","u" ),
            outputs=("x_1_hat", "x_2_hat","x_3_hat", "x_4_hat"),
        )
