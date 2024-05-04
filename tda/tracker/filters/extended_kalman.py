import numpy as np
from numpy.typing import NDArray
import scipy.linalg as la
from scipy.stats import multivariate_normal
from typing import Any, Callable, Dict, Tuple

from tda.common.measurement import Measurement

from .filter import Filter

class ExtendedKalman(Filter):
    f: Callable[[float, NDArray], NDArray]
    F: Callable[[float], NDArray]
    h: Callable[[float, NDArray], NDArray]
    H: Callable[[float], NDArray]
    Q: Callable[[float], NDArray]
    R: NDArray

    def __init__(self, x_hat_0, P_hat_0, f, F, h, H, Q, R):
        super().__init__(x_hat_0, P_hat_0)
        self.f = f  # nonlinear function propagating current state out by dt
        self.F = F  # linearization of f paramatarized by dt
        self.h = h  # nonlinear function taking a state vector and returning a meas vector
        self.H = H  # linearization of h
        self.Q = Q  # process noise covariance
        self.R = R  # measurement noise covariance

        assert self.f(0).shape == self.f(0).shape[1] \
            == self.F(0).shape[0] == self.F(0).shape[1] \
            == self.Q(0).shape[0] == self.Q(0).shape[1] == self.H(0).shape[1] \
            == self._num_states 
        
        assert self.R.shape[0] == self.R.shape[1] \
            == self.H(0).shape[0] == self.h(0).shape[0]
        
    
    def predict(self, time: float) -> Tuple[NDArray, NDArray]:
        dt = time - self.update_time
        F = self.F(dt)

        x_pre = self.f(dt, self.x_hat)
        P_pre = F @ self.P_hat @ F.T + self.Q(dt)

        return x_pre, P_pre
    

    def do_update(self, meas: Measurement) -> Tuple[NDArray, NDArray]:
        dt = meas.time - self.update_time
        H = self.H(dt)

        x_pred, P_pred = self.predict(meas.time)

        inov = meas.y - self.h(dt, self.x_hat)
        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T la.inv(S)
        self.x_hat = x_pred + K @ inov
        self.P_hat = (np.eye(self._num_states) - K @ H) @ P_pred

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
