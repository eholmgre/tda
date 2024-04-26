from abc import ABCMeta, abstractmethod
from typing import List
from numpy.typing import NDArray

from ..sensors.sensor import Sensor
from ..sim_engine import Simulation


class SimObject(metaclass=ABCMeta):
    state: NDArray  # must have dimension no less than 3

    _num_states: int  # conveinence variable for how many states we have
    _local_clock: float  # clock starting at this object's birth
    _payloads: List[Sensor]  # sensors this object may be carrying
    _sim: Simulation


    def __init__(self, initial_state: NDArray, simulation: Simulation):
        self._num_states = initial_state.shape[0]
        assert self._num_states >= 3
        self.state = initial_state
        self._local_clock = 0.0
        self._payloads = list()
        self._sim = simulation


    def add_payload(self, payload: Sensor):
        self._payloads.append(payload)


    @abstractmethod
    def pre_advance(self, ):
        pass
    

    @abstractmethod
    def advance(self, time_quanta: float):
        pass


    @abstractmethod
    def post_advance(self):
        pass

    @abstractmethod
    def _state_transition_function(self, dt:float, t:float=0.0):
        pass