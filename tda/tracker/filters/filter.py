from abc import ABCMeta, abstractmethod
from numpy.typing import NDArray
from typing import Dict, Tuple

from .history.filter_history import FilterHistory
from tda.common.measurement import Measurement


class Filter(metaclass=ABCMeta):
    def __init__(self, x_hat_0: NDArray, P0: NDArray) -> None:
        self._num_states = x_hat_0.shape[0]
        self.x_hat = x_hat_0
        self.P = P0

        self.history = FilterHistory(self, "")
        self.update_time = -1.0
        self.update_score = -1.0
        self.total_score = -1.0  # todo compute this


    @abstractmethod
    def predict(self, time: float) -> Tuple[NDArray, NDArray]:
        pass


    @abstractmethod
    def predict_meas(self, float) -> NDArray:
        pass


    def update(self, meas: Measurement) -> Tuple[NDArray, NDArray]:
        self.x_hat, self.P = self._do_update(meas)
        self.update_time = meas.time
        self.history.record()

        return self.x_hat, self.P, 
    

    def update_external(self, x_hat: NDArray, P: NDArray, time: float) -> None:
        self.x_hat = x_hat 
        self.P = P
        self.update_time = time
        self.history.record()
    

    @abstractmethod
    def compute_gain(self, time: float) -> NDArray:
        pass


    @abstractmethod
    def compute_S(self, time:float) -> NDArray:
        pass


    @abstractmethod
    def _do_update(self, meas: Measurement) -> Tuple[NDArray, NDArray]:
        pass


    @abstractmethod
    def meas_likelihood(self, meas: Measurement) -> float:
        pass


    @abstractmethod
    def meas_distance(self, meas: Measurement) -> float:
        pass

    
    @abstractmethod
    def get_position(self) -> Tuple[NDArray, NDArray]:
        pass
    

    @abstractmethod   
    def get_velocity(self) -> Tuple[NDArray, NDArray]:
        pass
    

    @abstractmethod
    def get_acceleration(self) -> Tuple[NDArray, NDArray]:
        pass


    def record(self) -> Dict:
        return self.history.save()
