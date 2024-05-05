import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple


from .filters.filter import Filter
from tda.common.measurement import Measurement


class Track():
    track_id: int
    filter: Filter
    meas_hist: List[Measurement]
    state_hist: List[NDArray]


    def __init__(self, track_id: int, track_filter: Filter):
        self.track_id = track_id
        self.filter = track_filter
        self.meas_hist = list()
        self.state_hist = list()

    
    def predict(self, time: float) -> Tuple[NDArray, NDArray]:
        return self.filter.predict(time)
    

    def meas_likelihood(self, meas: Measurement) -> float:
        return self.filter.meas_likelihood(meas)
    

    def update(self, meas: Measurement) -> Tuple[NDArray, NDArray]:
        return self.filter.update(meas)
    

    def get_state(self) -> NDArray:
        return self.filter.x_hat

    def get_uncert(self) -> float:
        return self.filter.P_hat.trace()
