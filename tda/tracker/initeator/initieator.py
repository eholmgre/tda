from abc import ABCMeta, abstractmethod
import logging
import numpy as np
from typing import List, Sequence

from ..filters.filter import Filter
from ..filters.linear_kalman import LinearKalman3, LinearKalman6, LinearKalman9
from ..track import Track
from ..tracker_param import TrackerParam
from tda.common.measurement import Measurement


track_id_ctr = 0


class Initeator(metaclass=ABCMeta):
    def __init__(self, param: TrackerParam) -> None:
        self.params = param


    def create_filter(self, meas: Measurement) -> Filter:
        P_0 = np.diag(self.params.filter_startQ)
        
        if self.params.filter_nstate == 3:
            x_0 = meas.y
            return LinearKalman3(x_0, P_0, self.params.filter_n3_q)
        
        if self.params.filter_nstate == 6:
            x_0 = np.zeros(6)
            x_0[0] = meas.y[0]
            x_0[2] = meas.y[1]
            x_0[4] = meas.y[2]

            return LinearKalman6(x_0, P_0, self.params.filter_n6_q)
        
        if self.params.filter_nstate == 9:
            x_0 = np.zeros(9)
            x_0[0] = meas.y[0]
            x_0[3] = meas.y[1]
            x_0[6] = meas.y[2]

            return LinearKalman9(x_0, P_0, self.params.filter_n9_q)
        
        logging.error(f"invalid filter type: {self.params.filter_nstate}")
        return None


    @abstractmethod
    def initeate_tracks(self, frame: Sequence[Measurement]) -> List[Track]:
        pass
