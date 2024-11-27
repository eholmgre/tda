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

        self.mu = np.array([0.7, 0.2, 0.1])
        self.Pi = np.array([[0.90, 0.08, 0.02],
                            [0.15, 0.70, 0.15],
                            [0.04, 0.16, 0.80]])
        


    def predict(self, time: float) -> Tuple[NDArray, NDArray]:
        pass
    

    def predict_meas(self, time: float) -> NDArray:
        pass
    
    def _get_R(self, meas):
        pass
    

    def _do_update(self, meas: Measurement) -> Tuple[NDArray, NDArray]:
        pass
    

    def compute_gain(self, time: float) -> NDArray:
        pass
    

    def compute_S(self, time: float) -> NDArray:
        pass
    

    def meas_likelihood(self, meas: Measurement) -> float:
        pass
    

    def meas_distance(self, meas: Measurement) -> float:
        passs


