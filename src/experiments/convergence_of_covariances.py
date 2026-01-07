from typing import Iterable
from src.utils import iterate as abstract_iter
import numpy as np
from numpy.typing import NDArray
import dataclasses

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


matplotlib.use("TkAgg")
plt.rcParams.update({"font.size": 7})

@dataclasses.dataclass
class ConvergenceCriterion:
    p: NDArray

    def __call__(self, p: NDArray) -> bool:
        condition = np.allclose(p , self.p, rtol=1e-05, atol=1e-08)
        self.p = p
        return condition

def discrete_lyapunov_function(f: NDArray, p: NDArray, q: NDArray) -> Iterable[NDArray]:
    def step(p_: NDArray) -> NDArray:
        return f @ p_ @ f.T + q

    done = ConvergenceCriterion(p)
    values = abstract_iter.converge(
        values=abstract_iter.iterate(step=step, start=p),
        done=done,
    )
    return values


def __generate_covariances(num: int) -> Iterable[NDArray]:
    pass


def __create_covariance_trajectories(covanriances: Iterable[NDArray]) -> list[Iterable[NDArray]]:
    ...

def plot_covariance_trajectories(trajectories: list[Iterable[NDArray]]) -> None:
    def flattened_trajectorues(_trajectories: list[Iterable[NDArray]]) -> Iterable[NDArray]:
        for item in _trajectories:
            item = np.array(item)
            item.reshape(item.shape[0], -1)
            yield item

    fig: Figure = plt.figure(figsize=(16, 9))
    ax:Axes = fig.add_subplot()
    for i, item in enumerate(flattened_trajectorues(trajectories)):
        for j in range(4):
            ax.plot(item[:, j], color=f"C{i}")

    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    f = np.array([
        [0.2, 0.4],
        [-0.4, 1]
    ])

    q = np.diag([1, 2])

    p = - 1* np.eye(2)

    # covariance_values = iter(discrete_lyapunov_function(f, p, q))
    covariance_values = np.array(list(discrete_lyapunov_function(f, p, q)))
    covariance_values = covariance_values.reshape((covariance_values.shape[0], -1))
    import matplotlib.pyplot as plt
    plt.figure(figsize=(16, 9))
    for i in range(4):
        plt.plot(covariance_values[:, i], color=f"C{i}")
    plt.tight_layout()
    plt.show()
