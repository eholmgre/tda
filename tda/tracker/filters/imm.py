import numpy as np
from numpy.typing import NDArray
import scipy.linalg as la
from typing import Tuple

from .filter import Filter
from .linear_kalman import LinearKalman6, LinearKalman9
from .history.imm_history import IMMHistory, LinearKalmanManauverHistory
from tda.common.measurement import Measurement


class LinearKalmanManuver(LinearKalman9):
    def __init__(self, x_hat_0: NDArray, P_0: NDArray, t0: float, q: float):
        super().__init__(x_hat_0, P_0, t0, q)

        self.update_omega: float = -1.0

        self.history = LinearKalmanManauverHistory(self)


    def omega(self) -> float:
        A = self.get_acceleration()[0]
        V = self.get_velocity()[0]

        if np.all(A == 0) or np.all(V == 0):
            o = 1
        else:
            o = la.norm(A) / la.norm(V)

        self.update_omega = o

        return o
    

    def F(self, dt: float) -> NDArray:
        omega = self.omega()
        iomega = 1 / omega
        iomegasq = 1 / omega ** 2

        sinodt = np.sin(omega * dt)
        cosodt = np.cos(omega * dt)

        F = np.zeros((9, 9))

        fma = np.array([[1, iomega * sinodt, iomegasq * (1 - cosodt)],
                        [0,          cosodt, iomega * sinodt        ],
                        [0, -omega * sinodt, cosodt                 ]])

        for i in range(3):
            a = 3 * i
            b = 3 * (i + 1)

            F[a:b, a:b] = fma

        return F


class IMM(Filter):
    def __init__(self, x_hat_0: NDArray, P_0: NDArray, t0: float, q_cv: float, q_ca: float, q_turn: float, mu_0: NDArray, Pi: NDArray):
        super().__init__(x_hat_0, P_0, t0)

        self.q_cv = q_cv
        self.q_ca = q_ca
        self.q_turn = q_turn

        x_hat_0_cv, P_0_cv = self.drop_accel(x_hat_0, P_0)

        self.cv_filter = LinearKalman6(x_hat_0_cv, P_0_cv, t0, q_cv)
        self.ca_filter = LinearKalman9(x_hat_0, P_0, t0, q_ca)
        self.manuver_filter = LinearKalmanManuver(x_hat_0, P_0, t0, q_turn)

        self.filters = [self.cv_filter, self.ca_filter, self.manuver_filter]

        self.mu = mu_0
        self.Pi = Pi

        self.history = IMMHistory(self)
        

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

    
    def drop_accel(self, x_ca: NDArray, P_ca: NDArray) -> Tuple[NDArray, NDArray]:
        """
        Drop acceleration terms from x and P for use in the cv filter.
        """
        x_cv = np.zeros(6)
        x_cv[::2] = x_ca[::3]
        x_cv[1::2] = x_ca[1::3]

        P_cv = np.zeros((6, 6))
        P_cv[::2, ::2] = P_ca[::3, ::3]
        P_cv[1::2, 1::2] = P_ca[1::3, 1::3]

        return x_cv, P_cv


    def _do_predict(self, time: float) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        dt = time - self.update_time

        ## grab last update posterior states for mixing in the imm
        x_cv = self.cv_filter.x_hat
        P_cv = self.cv_filter.P
        x_cv, P_cv = self.augment_cv(x_cv, P_cv)

        x_ca = self.ca_filter.x_hat
        P_ca = self.ca_filter.P

        x_ma = self.manuver_filter.x_hat
        P_ma = self.manuver_filter.P

        xs = np.array([x_cv, x_ca, x_ma])
        Ps = np.array([P_cv, P_ca, P_ma])

        ## compute chapman-kolomologrov transition probabilities
        c_bar = self.mu @ self.Pi
        omega = np.zeros((3, 3))
        
        for i in range(3):
            for j in range(3):
                omega[i, j] = (self.Pi[i, j] * self.mu[i]) / c_bar[j]

        ## compute chapman-kolomologrov mixed priors from transition probabilities & last cycle's posteriors
        x_mix = np.zeros((3,9))
        for i in range(3):
            for j in range(3):
                x_mix[i] += omega[j, i] * xs[j]  # omega i,j might be reversed

        P_mix = np.zeros((3, 9, 9))

        for i in range(3):
            for j in range(3):
                model_err = xs[j] - x_mix[j]
                P_mix[i] += omega[j, i] * (np.outer(model_err, model_err) + Ps[j])

        ## now that we have chapman-kolomologrov mixed priors, propagate with the filter's dynamics to get prediction priors
        # cv
        x_cv_mix, P_cv_mix = self.drop_accel(x_mix[0], P_mix[0])
        F_cv = self.cv_filter.F(dt)

        x_pre_cv = F_cv @ x_cv_mix
        P_pre_cv = F_cv @ P_cv_mix @ F_cv.T + self.cv_filter.Q(dt)

        x_pre_cv, P_pre_cv = self.augment_cv(x_pre_cv, P_pre_cv)

        # ca
        F_ca = self.ca_filter.F(dt)

        x_pre_ca = F_ca @ x_mix[1]
        P_pre_ca = F_ca @ P_mix[1] @ F_ca.T + self.ca_filter.Q(dt)

        # ma
        F_ma = self.manuver_filter.F(dt)  # this won't correctly count for omega from the mix

        x_pre_ma = F_ma @ x_mix[2]
        P_pre_ma = F_ma @ P_mix[2] @ F_ma.T + self.manuver_filter.Q(dt)

        x_prop = np.array([x_pre_cv, x_pre_ca, x_pre_ma])
        P_prop = np.array([P_pre_cv, P_pre_ca, P_pre_ma])

        x_pre = self.mu @ x_prop
        P_pre = np.zeros((9, 9))
        for i in range(3):
            P_pre += self.mu[i] * P_prop[i]

        return x_pre, P_pre, x_mix, P_mix
    
    def predict(self, time: float) -> Tuple[NDArray, NDArray]:
        x_pre, P_pre, _, _ = self._do_predict(time)

        return x_pre, P_pre
    

    def predict_meas(self, time: float) -> NDArray:
        x_pre, _ = self.predict(time)

        return self.ca_filter.H @ x_pre
        
    
    def _get_R(self, meas):
        return self.ca_filter._get_R(meas)
    

    def _do_update(self, meas: Measurement) -> Tuple[NDArray, NDArray]:
        x_pre, P_pre, x_mix, P_mix = self._do_predict(meas.time)

        # we need to do the updates with the mixed states from the prediction, so jam those into the filters
        last_time = self.update_time
        self.cv_filter.update_external(*self.drop_accel(x_mix[0], P_mix[0]), last_time)
        self.ca_filter.update_external(x_mix[1], P_mix[1], last_time)
        self.manuver_filter.update_external(x_mix[2], P_mix[2], last_time)

        # grab update likelihoods before changing filter state
        cv_likeli = self.cv_filter.meas_likelihood(meas)
        ca_likeli = self.ca_filter.meas_likelihood(meas)
        ma_likeli = self.manuver_filter.meas_likelihood(meas)
        likeli = np.array([cv_likeli, ca_likeli, ma_likeli])

        x_post_cv, P_post_cv = self.augment_cv(*self.cv_filter.update(meas))
        x_post_ca, P_post_ca = self.ca_filter.update(meas)
        x_post_ma, P_post_ma = self.manuver_filter.update(meas)

        xs = np.array([x_post_cv, x_post_ca, x_post_ma])
        Ps = np.array([P_post_cv, P_post_ca, P_post_ma])

        c_bar = self.mu @ self.Pi
        self.mu = c_bar * likeli  # element-wise mode prediction posterior
        self.mu /= self.mu.sum()

        x_post = self.mu @ xs

        P_post = np.zeros((9, 9))

        for i in range(3):
            filt_err = xs[i] - x_post
            P_post += self.mu[i] * (np.outer(filt_err, filt_err) + Ps[i])

        imm_inov = (self.ca_filter.H @ x_pre) - (self.ca_filter.H @ x_post)
        self.update_score = imm_inov @ self.ca_filter.H @ P_post @ self.ca_filter.H.T @ imm_inov

        return x_post, P_post        
    

    def compute_gain(self, time: float) -> NDArray:
        _, cv_gain = self.augment_cv(np.zeros(6), self.cv_filter.compute_gain(time))
        ca_gain = self.ca_filter.compute_gain(time)
        ma_gain = self.manuver_filter.compute_gain(time)

        return self.mu[0] * cv_gain + self.mu[1] * ca_gain + self.mu[2] * ma_gain
    

    def compute_S(self, time: float) -> NDArray:
        _, cv_S = self.augment_cv(np.zeros(6), self.cv_filter.compute_S(time))
        ca_S = self.ca_filter.compute_S(time)
        ma_S = self.manuver_filter.compute_S(time)

        return self.mu[0] * cv_S + self.mu[1] * ca_S + self.mu[2] * ma_S    


    def meas_likelihood(self, meas: Measurement) -> float:
        cv_likeli = self.cv_filter.meas_likelihood(meas)
        ca_likeli = self.ca_filter.meas_likelihood(meas)
        ma_likeli = self.manuver_filter.meas_likelihood(meas)

        return self.mu[0] * cv_likeli + self.mu[1] * ca_likeli + self.mu[2] * ma_likeli        


    def meas_distance(self, meas: Measurement) -> float:
        cv_dist = self.cv_filter.meas_distance(meas)
        ca_dist = self.ca_filter.meas_distance(meas)
        ma_dist = self.manuver_filter.meas_distance(meas)

        return self.mu[0] * cv_dist + self.mu[1] * ca_dist + self.mu[2] * ma_dist    


    def get_position(self) -> Tuple[NDArray, NDArray]:
        cv_pos = self.cv_filter.get_position()
        ca_pos = self.ca_filter.get_position()
        ma_pos = self.manuver_filter.get_position()

        imm_pos = self.mu[0] * cv_pos[0] + self.mu[1] * ca_pos[0] + self.mu[2] * ma_pos[0]
        imm_sig_pos = self.mu[0] * cv_pos[1] + self.mu[1] * ca_pos[1] + self.mu[2] * ma_pos[1]

        return imm_pos, imm_sig_pos
    

    def get_velocity(self) -> Tuple[NDArray, NDArray]:
        cv_vel = self.cv_filter.get_velocity()
        ca_vel = self.ca_filter.get_velocity()
        ma_vel = self.manuver_filter.get_velocity()

        imm_vel = self.mu[0] * cv_vel[0] + self.mu[1] * ca_vel[0] + self.mu[2] * ma_vel[0]
        imm_sig_vel = self.mu[0] * cv_vel[1] + self.mu[1] * ca_vel[1] + self.mu[2] * ma_vel[1]

        return imm_vel, imm_sig_vel


    def get_acceleration(self) -> Tuple[NDArray, NDArray]:
        cv_accel = self.cv_filter.get_acceleration()
        ca_accel = self.ca_filter.get_acceleration()
        ma_accel = self.manuver_filter.get_acceleration()

        imm_accel = self.mu[0] * cv_accel[0] + self.mu[1] * ca_accel[0] + self.mu[2] * ma_accel[0]
        imm_sig_accel = self.mu[0] * cv_accel[1] + self.mu[1] * ca_accel[1] + self.mu[2] * ma_accel[1]

        return imm_accel, imm_sig_accel