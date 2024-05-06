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


    def update(self, meas: Measurement) -> Tuple[NDArray, NDArray]:
        x_hat, P = self._do_update(meas)
        self.update_time = meas.time
        self._filter_history.append((self.update_time, x_hat, self.P))

        return x_hat, P


    @abstractmethod
    def _do_update(self, meas: Measurement) -> Tuple[NDArray, NDArray]:
        pass

    @abstractmethod
    def meas_likelihood(self, meas: Measurement) -> float:
        pass

    
    @abstractmethod
    def record(self) -> Dict[str, Any]:
        pass
