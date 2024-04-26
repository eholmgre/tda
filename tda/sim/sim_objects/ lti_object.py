from typing import Any, Dict, Sequence, Tuple
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


    def _do_advance(self, time_quanta):
        self.x = self._getF(time_quanta) @ self.x


    def is_done(self) -> bool:
        return False
    
   
    def _getF(self, dt: float) -> NDArray:
        F = dt * np.eye(self._num_states)

        if self._num_states >= 6:
            F[0, 3] = F[1, 4] = F[2, 5] = dt

        if self._num_states >= 9:
            F[3, 6] = F[4, 7] = F[5, 8] = dt
            F[0, 6] = F[1, 7] = F[2, 8] = (dt ** 2) / 2

        return F


    def record(self) -> Sequence[Tuple[str, Dict[str, Any]]]:
        records = list()
        for p in self._payloads:
            records.append(p.record())

        n = len(self._state_hist)
        record = dict()
        record["t"] = np.zeros(n)
        record["x"] = np.zeros(n)
        record["y"] = np.zeros(n)
        record["z"] = np.zeros(n)

        if self._num_states >= 6:
            record["xdot"] = np.zeros(n)
            record["ydot"] = np.zeros(n)
            record["zdot"] = np.zeros(n)

        if self._num_states >= 9:
            record["xdotdot"] = np.zeros(n)
            record["ydotdot"] = np.zeros(n)
            record["zdotdot"] = np.zeros(n)

        for i, (t, h) in enumerate(self._state_hist):
            record["t"][i] = t
            record["x"][i] = h[0]
            record["y"][i] = h[1]
            record["z"][i] = h[2]

            if self._num_states >= 6:
                record["xdot"][i] = h[3]
                record["ydot"][i] = h[4]
                record["zdot"][i] = h[5]

            if self._num_states >= 9:
                record["xdotdot"][i] = h[6]
                record["ydotdot"][i] = h[7]
                record["zdotdot"][i] = h[8]

            records.append((self.get_name(), record))

        return records
