import control
import numpy as np

from components.plant import ModelParametization, create_mass_spring_damper
from components.controller import Controller, FullStateLQRController

def create_simple_closed_loop_system(plant_params: ModelParametization):
    plant: control.StateSpace = create_mass_spring_damper(params=plant_params, full_state=True)
    controller = Controller(
        controller=FullStateLQRController(
            plant=plant,
            Q=10 * np.eye(4),
            R=10 * np.eye(1),
        ),
        inputs=("x_1","x_2","x_3","x_4", "ref"),
        outputs=("u", ),
    ).create_controller()

    print(plant)

    # return control.interconnect(
    #     [plant, controller],
    # )


if __name__ == "__main__":
    create_simple_closed_loop_system(plant_params=ModelParametization(
        m_1=10,
        m_2=5,
        d_1=0.9,
        d_2=0.9,
        k_1=0.4,
        k_2=0.4
    ))