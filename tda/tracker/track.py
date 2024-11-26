import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple

from .filters.filter import Filter
from tda.common.measurement import Measurement

class _StateHist():
    def __init__(self):
        self.state = []
        self.cov = []
        self.pos = []
        self.sig_pos = []
        self.vel = []
        self.sig_vel = []
        self.accel = []
        self.sig_accel = []
        self.score = []
        self.time = []

        self.num_updates = 0


    def record(self, filt: Filter):
        self.state.append(filt.x_hat)
        self.cov.append(filt.P)

        pos, sig_pos = filt.get_position()
        self.pos.append(pos)
        self.sig_pos.append(sig_pos)

        vel, sig_vel = filt.get_velocity()
        self.vel.append(vel)
        self.sig_vel.append(sig_vel)


        accel, sig_accel = filt.get_acceleration()
        self.accel.append(accel)
        self.sig_accel.append(sig_accel)

        self.score.append(filt.update_score)
        self.time.append(filt.update_time)

        self.num_updates += 1


class Track():
    def __init__(self, track_id: int, track_filter: Filter):
        self.track_id = track_id
        self.filter = track_filter
        self.meas_hist: List[Measurement]= list()
        self.state_hist: _StateHist =_StateHist()

    
    def predict(self, time: float) -> Tuple[NDArray, NDArray]:
        return self.filter.predict(time)
    

    def predict_meas(self, time: float) -> NDArray:
        return self.filter.predict_meas(time)
    

    def compute_gain(self, time: float) -> NDArray:
        return self.filter.compute_gain(time)
    

    def compute_S(self, time: float) -> NDArray:
        return self.filter.compute_S(time)
    

    def meas_likelihood(self, meas: Measurement) -> float:
        return self.filter.meas_likelihood(meas)
    

    def meas_distance(self, meas: Measurement) -> float:
        return self.filter.meas_distance(meas)
    

    def update(self, meas: Measurement) -> Tuple[NDArray, NDArray]:
        self.meas_hist.append(meas)
        x_hat, P = self.filter.update(meas)
        # x_hat[0: 3] += meas.sensor_pos
        self.state_hist.record(self.filter)
        return x_hat, P
    

    def update_external(self, x_hat: NDArray, P: NDArray, time: float) -> None:
        self.filter.update_external(x_hat, P, time)
        self.state_hist.record(self.filter)


    def get_state(self) -> NDArray:
        return self.filter.x_hat


    def get_uncert(self) -> float:
        return self.filter.P.trace()
    
    
    def __repr__(self) -> str:
        num_hits = len(self.meas_hist)
        last_score = self.state_hist.score[-1]
        avg_score = sum(self.state_hist.score) / self.state_hist.num_updates
        pos_cov = self.state_hist.sig_pos[-1]
        con95vol = 4 / 3 * np.pi * 2 * pos_cov[0] * pos_cov[1] * pos_cov[2]
        
        return f"Track {self.track_id} - num hits: {num_hits}, avg score: {avg_score}, last_score: {last_score} , 95% containment vol: {con95vol} m^3"
