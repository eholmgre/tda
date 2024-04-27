import numpy as np
from numpy.typing import NDArray
from scipy.stats import multivariate_normal
from typing import Any, Dict, Optional, Sequence, Tuple


from .sim_object import SimObject
from ..sim_engine import Simulation


class LTIObject(SimObject):
    Q: NDArray  # DT process noise covariance matrix
    F: Optional[NDArray]  # save the F matrix for speed

    def __init__(self,
                 object_id: int,
                 initial_state: NDArray,
                 simulation: Simulation,
                 Q: NDArray):
        super().__init__(object_id, initial_state, simulation)
        assert self._num_states % 3 == 0 
        
        self.Q = Q
        assert Q.shape[0] == self._num_states

        self.object_type = "lti_object"
        self.F = None
    

    def pre_advance(self):
        pass


    def _do_advance(self, time_quanta):
        if self.F is None:
            self.F = self._getF(time_quanta)

        self.state = self.F @ self.state + multivariate_normal.rvs(cov=self.Q)


    def is_done(self) -> bool:
        return False
    
   
    def _getF(self, dt: float) -> NDArray:
        F = np.eye(self._num_states, dtype=np.float64)

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
