import logging
import numpy as np
from numpy.typing import NDArray
import scipy.linalg as la
from scipy.stats import multivariate_normal
from typing import Any, Dict, Tuple

from .filter import Filter
from tda.common.measurement import Measurement


class LinearKalman3(Filter):
    def __init__(self, x_hat_0: NDArray, P_0: NDArray, q: float):
        super().__init__(x_hat_0, P_0)

        self.R: NDArray = np.zeros(3)

        self.H: NDArray = np.array([[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1]])
        
        self.q = q
        
    
    # x = [x, y, z]
    def F(self, dt: float) -> NDArray:
        return np.eye(3)

    
    def Q(self, dt) -> NDArray:
        Q = np.zeros((3, 3))
        # position Q
        Q[0, 0] = Q[1, 1] = Q[2, 2] = self.q * dt

        return Q


    def predict(self, time: float) -> Tuple[NDArray, NDArray]:
        dt = time - self.update_time
        F = self.F(dt)
        x_pre = F @ self.x_hat
        P_pre = F @ self.P @ F.T + self.Q(dt)

        return x_pre, P_pre
    

    def predict_meas(self, time: float) -> NDArray:
        x_pred, _ = self.predict(time)

        return self.H @ x_pred
    

    def _get_R(self, meas) -> NDArray:
        if meas.sensor_cov.trace() > 0:
            self.R = meas.sensor_cov
            return self.R
        elif self.R.trace() > 0:
            return self.R
        else:
            logging.warning("Measurement has no uncertanty")
            return np.eye(3)
    

    def _do_update(self, meas: Measurement) -> Tuple[NDArray, NDArray]:
        x_pred, P_pred = self.predict(meas.time)

        y_pred = self.H @ x_pred
        innov = meas.y - y_pred

        R = self._get_R(meas)

        S = self.H @ P_pred @ self.H.T + R
        S_inv = la.inv(S)
        self.update_score = innov @ S_inv @ innov
        K = P_pred @ self.H.T @ S_inv
        x_hat = x_pred + K @ innov
        # self.P_hat = (np.eye(self._num_states) - K @ self.H) @ P_pred
        
        # Joseph's Form cov update
        A = np.eye(self._num_states) - K @ self.H
        P = A @ P_pred @ A.T + K @ R @ K.T

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

        R = self._get_R(meas)

        z_hat = meas.y - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + R

        return multivariate_normal.pdf(z_hat, cov=S)
    

    def meas_distance(self, meas: Measurement) -> float:
        x_pred, P_pred = self.predict(meas.time)

        R = self._get_R(meas)

        z_hat = meas.y - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + R

        return z_hat @ la.inv(S) @ z_hat
    
    
    def get_position(self) -> Tuple[NDArray, NDArray]:
        return self.x_hat, np.sqrt(np.diag(self.P))
    
    
    def get_velocity(self) -> Tuple[NDArray, NDArray]:
        return np.zeros(3), np.ones(3) * np.inf
    

    def get_acceleration(self) -> Tuple[NDArray, NDArray]:
        return np.zeros(3), np.ones(3) * np.inf


class LinearKalman6(LinearKalman3):
    def __init__(self, x_hat_0: NDArray, P_0: NDArray, q: float):
        super().__init__(x_hat_0, P_0, q)

        self.H: NDArray = np.array([[1, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 0]])

    
    # x = [x, dx, y, dy, z, dz]
    def F(self, dt: float) -> NDArray:
        fca = np.array([[1, dt],
                        [0, 1]])
        
        F = np.zeros((6, 6))

        for i in range(3):
            a = 2 * i
            b = 2 * (i + 1)

            F[a : b, a : b] = fca

        return F

    
    def Q(self, dt) -> NDArray:
        qcv = self.q * dt * np.array([[(dt ** 2) / 3, dt / 2],
                                      [dt / 2,        1]])
        Q = np.zeros((6, 6))
        
        for i in range(3):
            a = 2 * i
            b = 2 * (i + 1)
            Q[a : b, a : b] = qcv

        return Q
    

    def get_position(self) -> Tuple[NDArray, NDArray]:
        return self.x_hat[::2], np.sqrt(np.diag(self.P)[::2])
    
    
    def get_velocity(self) -> Tuple[NDArray, NDArray]:
        return self.x_hat[1::2], np.sqrt(np.diag(self.P)[1::2])
    

    def get_acceleration(self) -> Tuple[NDArray, NDArray]:
        return np.zeros(3), np.ones(3) * np.inf


class LinearKalman9(LinearKalman6):
    def __init__(self, x_hat_0: NDArray, P_0: NDArray, q: float):
        super().__init__(x_hat_0, P_0, q)

        self.H: NDArray = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 1, 0, 0]])


    # x = [x, dx, ddx, y, dy, ddy, z, dz, ddz]
    def F(self, dt: float) -> NDArray:
        fca = np.array([[1, dt, (dt ** 2) / 2],
                        [0, 1, dt],
                        [0, 0, 1]])
        
        F = np.zeros((9, 9))

        for i in range(3):
            a = 3 * i
            b = 3 * (i + 1)

            F[a : b, a : b] = fca

        return F

    
    def Q(self, dt) -> NDArray:
        qcv = self.q * dt * np.array([[0, 0, 0],
                                      [0, 0, 0],
                                      [0, 0, 1]])
        Q = np.zeros((9, 9))
        
        for i in range(3):
            a = 3 * i
            b = 3 * (i + 1)
            Q[a : b, a : b] = qcv

        return Q


    def get_position(self) -> Tuple[NDArray, NDArray]:
        return self.x_hat[::3], np.sqrt(np.diag(self.P)[::3])
    
    
    def get_velocity(self) -> Tuple[NDArray, NDArray]:
        return self.x_hat[1::3], np.sqrt(np.diag(self.P)[1::3])
    

    def get_acceleration(self) -> Tuple[NDArray, NDArray]:
        return self.x_hat[2::3], np.sqrt(np.diag(self.P)[2::3])
