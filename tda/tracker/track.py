import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple

from .filters.filter import Filter
from tda.common.measurement import Measurement


class Track():
    def __init__(self, track_id: int, track_filter: Filter):
        self.track_id = track_id
        self.filter = track_filter
        self.meas_hist: List[Measurement]= list()
        self.state_hist: List[Tuple[NDArray, NDArray, float, float]]=list()

    
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
        x_hat, P, nis = self.filter.update(meas)
        # x_hat[0: 3] += meas.sensor_pos
        self.state_hist.append((x_hat, P, nis, meas.time))
        return x_hat, P
    

    def update_external(self, x_hat: NDArray, P: NDArray, time: float) -> None:
        self.filter.update_external(x_hat, P, time)
        self.state_hist.append((x_hat, P, 0, time))


    def get_state(self) -> NDArray:
        return self.filter.x_hat


    def get_uncert(self) -> float:
        return self.filter.P.trace()
    

    def get_state_hist(self, x_i: int, sigma: float=2.0) -> Tuple[NDArray, NDArray, NDArray]:
        n = len(self.state_hist)
        state = np.zeros(n)
        time = np.zeros_like(state)
        uncert = np.zeros((n, 2))

        for i, (x, P, nis, t) in enumerate(self.state_hist):
            state[i] = x[x_i]
            time[i] = t
            
            uncert_i = sigma * np.sqrt(P[x_i, x_i])
            uncert[i, 0] = x[x_i] - uncert_i
            uncert[i, 1] = x[x_i] + uncert_i

        return state, uncert, time
    
    def __repr__(self) -> str:
        num_hits = len(self.meas_hist)
        last_score = self.state_hist[-1][2]
        avg_score = sum([hist[2] for hist in self.state_hist]) / num_hits

        pos_cov = np.diag(self.state_hist[-1][1])[::3]
        con95vol = 4 / 3 * np.pi * 2 * pos_cov[0] * pos_cov[1] * pos_cov[2]
        
        return f"Track {self.track_id} - num hits: {num_hits}, avg score: {avg_score}, last_score: {last_score}, 95% containment vol: {con95vol} m^3"
