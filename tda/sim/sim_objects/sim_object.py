from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Sequence, Tuple
from numpy.typing import NDArray

from ..sensors.sensor import Sensor
from ..sim_engine import Simulation


class SimObject(metaclass=ABCMeta):
    state: NDArray  # must have dimension no less than 3 (x, y, z)

    _num_states: int  # conveinence variable for how many states we have
    _local_clock: float  # clock starting at this object's birth
    _payloads: List[Sensor]  # sensors this object may be carrying
    _sim: Simulation  # refrence to simulation
    _state_hist: List[Tuple[float, NDArray]]  # state hist w/ sim times


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
    def pre_advance(self):
        pass
    

    def advance(self):
        dt = self._sim._time_delta
        self._do_advance(dt)
        self._local_clock += dt


    @abstractmethod
    def _do_advance(self, time_quanta: float):
        pass


    def post_advance(self):
        for p in self._payloads:
            _sim.meas_queue.append(p.create_measurements())

        self._state_hist.append((_sim._sim_time, self.state))
        

    @abstractmethod
    def is_done(self) -> bool:
        pass

    @abstractmethod
    def record(self) -> Sequance[Tuple[str, Dict[str, Any]]]:
        pass