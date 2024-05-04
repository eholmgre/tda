import numpy as np
import matplotlib.pyplot as plt
from tda.sim.sim_engine import Simulation
from tda.sim.sim_objects.lti_object import LTIObject
from tda.sim.sensors import clutter_model
from tda.sim.sensors.radar import Radar

from tda.tracker.filters.extended_kalman import ExtendedKalman


def get_F(dt: float):
    F = np.eye(6, dtype=np.float64)
    F[0, 3] = F[1, 4] = F[2, 5] = dt
    #F[3, 6] = F[4, 7] = F[5, 8] = dt
    #F[0, 6] = F[1, 7] = F[2, 8] = (dt ** 2) / 2

    return F


def f(dt, x):
    return get_F(dt) @ x


def h(x):
    rho = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    the = np.arctan2(x[1], x[0])
    psi = np.arccos(x[2] / rho)
    
    return np.array([the, psi, rho])


def get_H(X):
    x = X[0]
    y = X[1]
    z = X[2]
    
    return np.array([[-y/(x**2 + y**2), x/(x**2 + y**2), 0, 0, 0, 0],
                     [(x*z)/((1 - z**2/(x**2 + y**2 + z**2))**(1/2)*(x**2 + y**2 + z**2)**(3/2)), (y*z)/((1 - z**2/(x**2 + y**2 + z**2))**(1/2)*(x**2 + y**2 + z**2)**(3/2)), -(1/(x**2 + y**2 + z**2)**(1/2) - z**2/(x**2 + y**2 + z**2)**(3/2))/(1 - z**2/(x**2 + y**2 + z**2))**(1/2), 0, 0, 0],
                     [x/(x**2 + y**2 + z**2)**(1/2), y/(x**2 + y**2 + z**2)**(1/2), z/(x**2 + y**2 + z**2)**(1/2), 0, 0, 0]
                    ])


def Q(dt: float):
    Q = np.zeros((6, 6))
    Q[0, 0] = 0.35
    Q[1, 1] = 0.35

    return dt * Q


if __name__ == "__main__":
    sim = Simulation()

    R = np.array([[np.pi / 180, 0, 0],
                [0, np.pi / 180, 0],
                [0, 0, 10]])

    platform = LTIObject(1, np.array([0, 50, 0]), sim, np.zeros((3, 3)))
    radar = Radar(1, platform, 5.0, R)
    platform.add_payload(radar)

    target_Q = np.zeros((6, 6))
    target_Q[0, 0] = 0.25
    target_Q[1, 1] = 0.33
    target_Q[2, 2] = 1e-6
    target_Q[3, 3] = 1e-6
    target_Q[4, 4] = 1e-6
    target_Q[5, 5] = 1e-6

    target = LTIObject(2, np.array([10, 12, 0, 1, 2, 0]), sim, target_Q)

    sim._sim_objects.extend([platform, target])
    sim_records = sim.run()

    x_hat_0 = np.ones(6)
    P_hat_0 = 1e9 * np.eye(6)

    kf = ExtendedKalman(x_hat_0, P_hat_0, f, get_F, h, get_H, Q, R)

    for frame in sim.meas_queue:
        for m in frame:
            kf.update(m)

    kf_record = kf.record()
