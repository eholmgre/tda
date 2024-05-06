import numpy as np
from numpy.typing import NDArray
import scipy.linalg as la
from scipy.stats import multivariate_normal
from typing import Any, Callable, Dict, Tuple

from .filter import Filter
from tda.common.measurement import Measurement


class ExtendedKalman(Filter):
    def __init__(self, x_hat_0, P_hat_0, f, F, h, H, Q, R):
        super().__init__(x_hat_0, P_hat_0)
        self.f = f  # nonlinear function propagating current state out by dt
        self.F = F  # linearization of f paramatarized by dt
        self.h = h  # nonlinear function taking a state vector and returning a meas vector
        self.H = H  # linearization of h
        self.Q = Q  # process noise covariance
        self.R = R  # measurement noise covariance

        assert self.f(1.0, self.x_hat).shape[0] \
            == self.F(1.0).shape[0] == self.F(1.0).shape[1] \
            == self.Q(1.0).shape[0] == self.Q(1.0).shape[1] == self.H(self.x_hat).shape[1] \
            == self._num_states 
        
        assert self.R.shape[0] == self.R.shape[1] \
            == self.H(self.x_hat).shape[0] == self.h(self.x_hat).shape[0]
        
    
    def predict(self, time: float) -> Tuple[NDArray, NDArray]:
        dt = time - self.update_time
        F = self.F(dt)

        x_pre = self.f(dt, self.x_hat)
        P_pre = F @ self.P @ F.T + self.Q(dt)

        return x_pre, P_pre
    

    def _do_update(self, meas: Measurement) -> Tuple[NDArray, NDArray]:
        x_pred, P_pred = self.predict(meas.time)
        H = self.H(x_pred)

        inov = meas.y - self.h(x_pred)
        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ la.inv(S)
        self.x_hat = x_pred + K @ inov
        self.P_hat = (np.eye(self._num_states) - K @ H) @ P_pred

        return self.x_hat, self.P


    def meas_likelihood(self, meas: Measurement) -> float:
        x_pred, P_pred = self.predict(meas.time)
        H = self.H(x_pred)

        innov = meas.y - self.h(x_pred)
        S = H @ P_pred @ H.T + self.R

        return -1 * np.log(multivariate_normal.pdf(innov, cov=S))


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
