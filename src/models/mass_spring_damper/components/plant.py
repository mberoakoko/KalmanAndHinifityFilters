import control
import numpy as np
from numpy.typing import NDArray
from src.models.mass_spring_damper.config.model_parametization import ModelParametization

def create_mass_spring_damper(params: ModelParametization, full_state: bool = False)-> control.StateSpace:
    A: NDArray= np.array([
        [0, 1, 0, 0],
        [-(params.k_1+params.k_2)/params.m_1, -(params.d_2 - params.d_1)/params.m_1, params.k_2/params.m_1, params.d_1/params.m_1],
        [0, 0, 0, 1],
        [params.k_2/params.m_2, params.d_2/params.m_1, -params.k_2/params.m_1, -params.d_2/params.m_1],
    ])
    B: NDArray= np.array([
        [0],
        [0],
        [0],
        [1/params.m_2],
    ])

    C: NDArray = np.array([1, 0, 0, 0]) if not full_state else np.eye(4)

    D: NDArray =np.zeros(1) if not full_state else np.zeros((4, 1))

    return control.ss(A, B, C, D,
                      inputs=("u"),
                      outputs=("x_1") if not full_state else ("x_1", "x_2", "x_3", "x_4")
                    )
