import numpy as np
from numpy.typing import NDArray
import scipy.linalg as la
from scipy.stats import multivariate_normal
from typing import Any, Callable, Dict, Tuple

from .filter import Filter
from tda.common.measurement import Measurement


class LinearKalman(Filter):
    def __init__(self, x_hat_0: NDArray, P_0: NDArray, F: Callable[[float], NDArray],
                 H: NDArray, Q: Callable[[float], NDArray], R: NDArray):
        super().__init__(x_hat_0, P_0)
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
        P_pre = F @ self.P @ F.T + self.Q(dt)

        return x_pre, P_pre
    

    def predict_meas(self, time: float) -> NDArray:
        x_pred, _ = self.predict(time)

        return self.H @ x_pred
    

    def _do_update(self, meas: Measurement) -> Tuple[NDArray, NDArray]:
        x_pred, P_pred = self.predict(meas.time)

        innov = meas.y - self.H @ x_pred

        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ la.inv(S)
        x_hat = x_pred + K @ innov
        # self.P_hat = (np.eye(self._num_states) - K @ self.H) @ P_pred
        
        # Joseph's Form cov update
        A = np.eye(self._num_states) - K @ self.H
        P = A @ P_pred @ A.T + K @ self.R @ K.T

        return x_hat, P
    

    def compute_gain(self, time: float) -> NDArray:
        _, P_pred = self.predict(time)
        S = self.H @ P_pred @ self.H.T + self.R

        return  P_pred @ self.H.T @ la.inv(S)
    

    def compute_S(self, time: float) -> NDArray:
        _, P_pred = self.predict(time)
        return self.H @ P_pred @ self.H.T + self.R
    

    def meas_likelihood(self, meas: Measurement) -> float:
        x_pred, P_pred = self.predict(meas.time)

        z_hat = meas.y - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R

        return multivariate_normal.pdf(meas.y, mean=z_hat, cov=S)
    

    def meas_distance(self, meas: Measurement) -> float:
        x_pred, P_pred = self.predict(meas.time)

        z_hat = meas.y - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R

        return z_hat @ la.inv(S) @ z_hat


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
