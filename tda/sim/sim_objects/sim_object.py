from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Sequence, Tuple
from numpy.typing import NDArray

from ..sensors.sensor import Sensor
from ..sim_engine import Simulation


class SimObject(metaclass=ABCMeta):
    state: NDArray  # must have dimension no less than 3 (x, y, z)
    object_id: int
    object_type: str

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


    def advance(self):
        dt = self._sim._time_delta
        self._do_advance(dt)
        self._local_clock += dt


    def post_advance(self):
        for p in self._payloads:
            self._sim.meas_queue.append(p.create_measurements())

        self._state_hist.append((self._sim._sim_time, self.state))

    
    def get_name(self) -> str:
        return f"{self.object_type}{self.object_id}"


    @abstractmethod
    def pre_advance(self):
        "object specific logic to prepare to update to the current state"
        pass
    

    @abstractmethod
    def _do_advance(self, time_quanta: float):
        "object specific logic to update its state to the current step"
        pass


    @abstractmethod
    def is_done(self) -> bool:
        "whether the object has finished its role in the engagement"
        pass


    @abstractmethod
    def record(self) -> Sequence[Tuple[str, Dict[str, Any]]]:
        "returns a list holding the record of the object and the records of all of its payloads"
        pass
