import numpy as np
from numpy.typing import NDArray
import scipy.linalg as la
from scipy.stats import multivariate_normal
from typing import Any, Callable, Dict, Tuple

from tda.common.measurement import Measurement

from .filter import Filter

class LinearKalman(Filter):
    F: Callable[[float], NDArray]  # state transition matrix (paramatarized by time delta)
    H: NDArray  # measurement model matrix
    Q: Callable[[float], NDArray]  # process noise covariance matrix (paramatraized by time)
    R: NDArray  # measurement noise covariance matrix


    def __init__(self, x_hat_0, P_hat_0, F, H, Q, R):
        super().__init__(x_hat_0, P_hat_0)
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R

        assert self.F(0).shape[0] == self.F(0).shape[1] \
            == self.Q(0).shape[0] == self.Q(0).shape[1] == self.H.shape[1] \
            == self._num_states 
        
        assert self.R.shape[0] == self.R.shape[1] == self.H.shape[0]
    

    def predict(self, time: float) -> Tuple[NDArray, NDArray]:
        dt = time - self.update_time
        F = self.F(dt)
        x_pre = F @ self.x_hat
        P_pre = F @ self.P_hat @ F.T + self.Q(dt)

        return x_pre, P_pre
    

    def do_update(self, meas: Measurement) -> Tuple[NDArray, NDArray]:
        x_pred, P_pred = self.predict(meas.time)

        K = P_pred @ self.H.T @ la.inv(self.H @ P_pred @ self.H.T + self.R)
        self.x_hat = x_pred + K @ (meas.y - self.H @ x_pred)
        self.P_hat = (np.eye(self._num_states) - K @ self.H) @ P_pred

        return self.x_hat, self.P_hat
    

    def meas_likelihood(self, meas: Measurement) -> float:
        x_pred, P_pred = self.predict(meas.time)

        innov = meas.y - self.H @ x_pred
        P_y_pred = self.H @ P_pred @ self.H.T

        return -1 * np.log(multivariate_normal.pdf(innov, cov=P_y_pred))


    def record(self) -> Dict[str, Any]:
        r = dict()
        n = len(self._filter_history)

        r["t"] = np.zeros(n)
        r["x_hat"] = np.zeros((n, self._num_states))
        r["P_hat"] = np.zeros((n, self._num_states, self._num_states))

        for i, (t, x, p) in enumerate(self._filter_history):
            r["t"][i] = t
            r["x_hat"][i] = x
            r["P_hat"][i] = p
        
        return r


