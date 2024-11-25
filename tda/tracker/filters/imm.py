import numpy as np
from numpy.typing import NDArray
import scipy.linalg as la
from scipy.stats import multivariate_normal
from typing import Any, Callable, Dict, Tuple

from .filter import Filter,
from .linear_kalman import LinearKalman
from tda.common.measurement import Measurement


class IMM(Filter):
    def __init__(self, x_hat_0: NDArray, P_0: NDArray):
        super().__init__(x_hat_0, P_0)

        self.filters = []


        # x = [x, xdot, y, ydot, z, zdot]
        F1 = lambda dt: np.array([[1, dt, 0, 0,  0, 0 ],
                                  [0, 1,  0, 0,  0, 0 ],
                                  [0, 0,  1, dt, 0, 0 ],
                                  [0, 0,  0, 1,  0, 0 ],
                                  [0, 0,  0, 0,  1, dt],
                                  [0, 0,  0, 0,  0, 1 ]])
        
        H1 = np.array([[1, 0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0]])
        
        def Q1(dt):
            qcv = np.array([])
            Q = np.zeros(6)
            

        R = np.zeros(3)

        steady_state_filter = LinearKalman(x_hat_0, P_0, F1, H1, Q1, R)
    

    def predict(self, time: float) -> Tuple[NDArray, NDArray]:
        dt = time - self.update_time
        F = self.F(dt)
        x_pre = F @ self.x_hat
        P_pre = F @ self.P @ F.T + self.Q(dt)

        return x_pre, P_pre
    

    def predict_meas(self, time: float) -> NDArray:
        x_pred, _ = self.predict(time)

        return self.H @ x_pred
    
    def _get_R(self, meas):
        if meas.sensor_cov.trace() > 0:
            R = meas.sensor_cov
        else:
            R = self.R

        return R
    

    def _do_update(self, meas: Measurement) -> Tuple[NDArray, NDArray, float]:
        x_pred, P_pred = self.predict(meas.time)

        y_pred = self.H @ x_pred
        innov = meas.y - y_pred

        R = self._get_R(meas)

        S = self.H @ P_pred @ self.H.T + R
        S_inv = la.inv(S)
        nis = innov @ S_inv @ innov
        K = P_pred @ self.H.T @ S_inv
        x_hat = x_pred + K @ innov
        # self.P_hat = (np.eye(self._num_states) - K @ self.H) @ P_pred
        
        # Joseph's Form cov update
        A = np.eye(self._num_states) - K @ self.H
        P = A @ P_pred @ A.T + K @ R @ K.T

        return x_hat, P, nis
    

    def compute_gain(self, time: float) -> NDArray:
        _, P_pred = self.predict(time)
        S = self.H @ P_pred @ self.H.T + self.R

        return  P_pred @ self.H.T @ la.inv(S)
    

    def compute_S(self, time: float) -> NDArray:
        _, P_pred = self.predict(time)
        return self.H @ P_pred @ self.H.T + self.R
    

    def meas_likelihood(self, meas: Measurement) -> float:
        x_pred, P_pred = self.predict(meas.time)

        R = self._get_R(meas)

        z_hat = meas.y - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + R

        return multivariate_normal.pdf(meas.y, mean=z_hat, cov=S)
    

    def meas_distance(self, meas: Measurement) -> float:
        x_pred, P_pred = self.predict(meas.time)

        R = self._get_R(meas)

        z_hat = meas.y - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + R

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
