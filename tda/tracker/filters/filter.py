from abc import ABCMeta, abstractmethod
from numpy.typing import NDArray
from typing import Any, Dict, List, Tuple

from tda.common.measurement import Measurement


class Filter(metaclass=ABCMeta):
    def __init__(self, x_hat_0: NDArray, P0: NDArray) -> None:
        self._num_states = x_hat_0.shape[0]
        self.x_hat = x_hat_0
        self.P = P0

        self.update_time = -1.0
        self._filter_history: List[Tuple[float, NDArray, NDArray]] = list()


    @abstractmethod
    def predict(self, time: float) -> Tuple[NDArray, NDArray]:
        pass


    @abstractmethod
    def predict_meas(self, float) -> NDArray:
        pass


    def update(self, meas: Measurement) -> Tuple[NDArray, NDArray, float]:
        self.x_hat, self.P, nis = self._do_update(meas)
        self.update_time = meas.time
        self._filter_history.append((self.update_time, self.x_hat, self.P))

        return self.x_hat, self.P, nis
    

    def update_external(self, x_hat: NDArray, P: NDArray, time: float):
        self.x_hat = x_hat 
        self.P = P
        self.update_time = time
        self._filter_history.append((self.update_time, self.x_hat, self.P))
    

    @abstractmethod
    def compute_gain(self, time: float) -> NDArray:
        pass


    @abstractmethod
    def compute_S(self, time:float) -> NDArray:
        pass


    @abstractmethod
    def _do_update(self, meas: Measurement) -> Tuple[NDArray, NDArray, float]:
        pass


    @abstractmethod
    def meas_likelihood(self, meas: Measurement) -> float:
        pass


    @abstractmethod
    def meas_distance(self, meas: Measurement) -> float:
        pass

    
    @abstractmethod
    def record(self) -> Dict[str, Any]:
        pass
