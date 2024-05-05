import numpy as np
from numpy.typing import NDArray

from .filters.linear_kalman import LinearKalman

track_id_ctr: int=1

def F(dt: float):
    F = np.eye(6, dtype=np.float64)
    F[0, 3] = F[1, 4] = F[2, 5] = dt
    #F[3, 6] = F[4, 7] = F[5, 8] = dt
    #F[0, 6] = F[1, 7] = F[2, 8] = (dt ** 2) / 2

    return F

def Q(dt: float):
    Q = np.zeros((6, 6))
    Q[0, 0] = 0.35
    Q[1, 1] = 0.35

    return dt * Q

R = 3 * np.eye(3)

H = np.array([[1, 0, 0, 1, 0, 0],
              [0, 1, 0, 0, 1, 0],
              [0, 0, 1, 0, 0, 1]])

def hinv(y: NDArray) -> NDArray:
    return np.array([y[0], y[1], y[2], 0, 0, 0])

def create_linear_filter(meas) -> LinearKalman:
    x_hat_0 = hinv(meas.y)
    P_hat_0 = np.eye(6) * 1e9

    return LinearKalman(x_hat_0, P_hat_0, F, H, Q, R)
