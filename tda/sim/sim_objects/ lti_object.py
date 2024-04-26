import numpy as np
from numpy.typing import NDArray

from .sim_object import SimObject
from ..sim_engine import Simulation


class LTIObject(SimObject):
    F: NDArray  # DT state transition matrix
    G: NDArray  # control mapping matrix
    u: NDArray  # control input vector
    W: NDArray  # DT process noise covariance matrix


    def __init__(self,
                 initial_state: NDArray,
                 simulation: Simulation,
                 W: NDArray):
        super().__init__(initial_state, simulation)
        assert self._num_states % 3 == 0 
        
        self.W = W
        assert W.shape[0] == self._num_states
    

    def pre_advance(self):
        pass


    def advance(self, time_quanta):
        self.x = self._getF(time_quanta) @ self.x


    def _getF(self, dt: float) -> NDArray:
        F = dt * np.eye(self._num_states)

        if self._num_states >= 6:
            F[0, 3] = F[1, 4] = F[2, 5] = dt

        if self._num_states >= 9:
            F[3, 6] = F[4, 7] = F[5, 8] = dt
            F[0, 6] = F[1, 7] = F[2, 8] = (dt ** 2) / 2

        return F
