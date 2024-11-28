import numpy as np
from numpy.typing import NDArray
import scipy.linalg as la
from scipy.stats import multivariate_normal
from typing import Any, Callable, Dict, Tuple

from .filter import Filter
from .linear_kalman import LinearKalman6, LinearKalman9
from tda.common.measurement import Measurement


class LinearKalmanManuver(LinearKalman9):
    def __init__(self, x_hat_0: NDArray, P_0: NDArray, q: float):
        super().__init__(x_hat_0, P_0, q)


    def omega(self) -> float:
        return la.norm(self.get_acceleration()[0]) / la.norm(self.get_position()[0])
    

    def F(self, dt: float) -> NDArray:
        omega = self.omega()
        iomega = 1 / omega
        iomegasq = 1 / omega ** 2

        sinodt = np.sin(omega * dt)
        cosodt = np.cos(omega * dt)

        return np.array([[1, iomega * sinodt, iomegasq * (1 - cosodt)],
                         [0,          cosodt, iomega * sinodt        ],
                         [0, -omega * sinodt, cosodt                 ]])


class IMM(Filter):
    def __init__(self, x_hat_0: NDArray, P_0: NDArray, q_cv: float, q_ca: float, q_turn: float):
        super().__init__(x_hat_0, P_0)

        self.q_cv = q_cv
        self.q_ca = q_ca
        self.q_turn = q_turn

        self.cv_filter = LinearKalman6(x_hat_0, P_0, q_cv)
        self.ca_filter = LinearKalman9(x_hat_0, P_0, q_ca)
        self.manuver_filter = LinearKalmanManuver(x_hat_0, P_0, q_turn)

        self.filters = [self.cv_filter, self.ca_filter, self.manuver_filter]

        self.mu = np.array([0.7, 0.2, 0.1])
        
        self.Pi = np.array([[0.90, 0.08, 0.02],
                            [0.15, 0.70, 0.15],
                            [0.04, 0.16, 0.80]])


    def augment_cv(self, x_cv: NDArray, P_cv:NDArray) -> Tuple[NDArray, NDArray]:
        """
        We need to transform the cv filter state to a ca state.
        This seems biassed, but Bar-Shalom says it's ok?
        (Estimation w/ apps to tracking & nav pg 470)
        cv state vector: [x, dx,      y, dy,      z, dz     ]
        ca state vector: [x, dx, ddx, y, dy, ddy, z, dz, ddz]
        """
        x_ca = np.zeros(9)
        x_ca[::3] = x_cv[::2]
        x_ca[1::3] = x_cv[1::2]
        
        P_ca = np.zeros((9, 9))
        P_ca[::3, ::3] = P_cv[::2, ::2]
        P_ca[1::3, 1::3] = P_cv[1::2, 1::2]

        return x_ca, P_ca


    def predict(self, time: float) -> Tuple[NDArray, NDArray]:
        x_cv_pre, P_cv_pre = self.augment_cv(*self.cv_filter.predict(time))
        x_ca_pre, P_ca_pre = self.ca_filter.predict(time)
        x_ma_pre, P_ma_pre = self.manuver_filter.predict(time)

        xs = np.array([x_cv_pre, x_ca_pre, x_ma_pre])

        c_bar = self.mu @ self.Pi
        omega = np.zeros((3, 3))
        
        for i in range(3):
            for j in range(3):
                omega[i, j] = (self.Pi[i, j] * self.mu[i]) / c_bar[j]

        x_pre = omega @ xs

        Ps = np.array([P_cv_pre, P_ca_pre, P_ma_pre])
        P_pre = np.zeros((9, 9))

        for i in range(3):
            model_err = xs[i] - x_pre
            P_pre += omega[i] * (np.outer(model_err, model_err) + Ps[i])

        return x_pre, P_pre
    

    def predict_meas(self, time: float) -> NDArray:
        x_pre, _ = self.predict(time)

        return self.ca_filter.H @ x_pre
        
    
    def _get_R(self, meas):
        return self.ca_filter._get_R(meas)
    

    def _do_update(self, meas: Measurement) -> Tuple[NDArray, NDArray]:
        x_pre, P_pre = self.predict(meas.time)

        xs = np.zeros((3, 9))
        Ps = np.zeros((3, 9, 9))
        likeli = np.zeros(3)

        for i in range(3):
            filt_x, filt_P = self.filters[i].update(meas)
            xs[i] = filt_x
            Ps[i] = filt_P
            likeli[i] = self.filters[i].meas_likelihood(meas)

        mu = c_bar * likeli  # element-wise mode prediction posterior
        self.mu /= mu.sum()

        x_post = self.mu @ xs

        P_post = np.zeros((9, 9))

        for i in range(3):
            filt_err = self.filters[i].x_hat - x_post
            P_post += mu[i] * (np.outer(filt_err, filt_err) + Ps[i])


        return x_post, P_post        
    

    def compute_gain(self, time: float) -> NDArray:
        return self.ca_filter.compute_gain(time)
    

    def compute_S(self, time: float) -> NDArray:
        return self.ca_filter.compute_S(time)
    

    def meas_likelihood(self, meas: Measurement) -> float:
        return self.ca_filter.meas_likelihood(meas)
    

    def meas_distance(self, meas: Measurement) -> float:
        return self.ca_filter.meas_distance(meas)