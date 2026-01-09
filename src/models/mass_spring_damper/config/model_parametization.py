import dataclasses

@dataclasses.dataclass(frozen=True)
class ModelParametization:
    m_1: float
    m_2: float
    k_1: float
    k_2: float
    d_1: float
    d_2: float
