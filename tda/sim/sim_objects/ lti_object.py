import numpy as np
from numpy.typing import NDArray

from .sim_object import SimObject
from ..sim_engine import Simulation


class LTIObject(SimObject):
    def __init__(self,
                 initial_state: NDArray,
                 simulation: Simulation,
                 dt: float,
                 A: NDArray,
                 B: NDArray,
                 u: NDArray,
                 Gamma: NDArray,
                 W: NDArray
    ):
        super().__init__(initial_state, simulation)