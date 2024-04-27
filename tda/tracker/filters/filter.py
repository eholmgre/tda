from abc import ABCMeta, abstractmethod
from numpy.typing import NDArray
from typing import Any, Dict, List, Tuple

from tda.common.measurement import Measurement


class Filter(metaclass=ABCMeta):
    x_hat: NDArray  # estimated state (posterior from last update)
    P_hat: NDArray  # estimated cov (posterior from last update
    update_time: float

    _num_states: int  # conveinence variable for how many states we have

    _filter_history: List[Tuple[float, NDArray, NDArray]]


    def __init__(self, x_hat_0, P_hat_0):
        self._num_states = x_hat_0.shape[0]
        self.x_hat = x_hat_0
        self.P_hat = P_hat_0

        self.update_time = -1.0
        self._filter_history = list()


    @abstractmethod
    def predict(self, time: float) -> Tuple[NDArray, NDArray]:
        pass


    def update(self, meas: Measurement):
        self.do_update(meas)
        self.update_time = meas.time
        self._filter_history.append((self.update_time, self.x_hat, self.P_hat))


    @abstractmethod
    def do_update(self, meas: Measurement) -> Tuple[NDArray, NDArray]:
        pass

    
    @abstractmethod
    def record(self) -> Dict[str, Any]:
        pass