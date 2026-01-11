from typing import overload, override

from numpy.typing import NDArray
import control
import dataclasses
import abc

class IControllerStrategy(abc.ABC):

    @abc.abstractmethod
    def update(self, t: float, x: NDArray, u: NDArray, params:dict = None) -> NDArray | None:
        raise NotImplementedError

    def output(self, t: float, x: NDArray, u: NDArray, params:dict = None) -> NDArray | None:
        raise NotImplementedError


@dataclasses.dataclass
class FullStateLQRController(IControllerStrategy):
    plant: control.StateSpace
    Q: NDArray
    R: NDArray
    K: NDArray = dataclasses.field(init=False)

    def __post_init__(self):
        K, _, _ = control.lqr(self.plant.A, self.plant.B, self.Q, self.R)

    @override
    def update(self, t: float, x: NDArray, u: NDArray, params:dict = None) -> NDArray | None:
        return None

    @override
    def output(self, t: float, x: NDArray, u: NDArray, params:dict = None) -> NDArray | None:
        return self.K @ x



@dataclasses.dataclass
class Controller:
    controller: IControllerStrategy
    inputs: tuple[str]
    outputs: tuple[str]
    name: str = "Controller"

    def create_controller(self) -> control.NonlinearIOSystem:
            return control.NonlinearIOSystem(
                self.controller.update, self.controller.output,
                name=self.name,
                inputs=self.inputs,
                outputs=self.outputs
            )



