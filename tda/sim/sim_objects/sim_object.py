from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Sequence, Tuple
from numpy.typing import NDArray

from ..ground_truth import GroundTruth
from ..sensors.sensor import Sensor


class SimObject(metaclass=ABCMeta):
    def __init__(self, object_id: int, object_type: str,
                 initial_state: NDArray, simulation: "Simulation") -> None:
        self.object_id = object_id
        self.object_type = object_type
        self._num_states = initial_state.shape[0]
        assert self._num_states >= 3
        self.state = initial_state
        self._sim = simulation

        self._local_clock = 0.0
        self._payloads: List[Sensor] = list()
        self._state_hist: List[Tuple[float, NDArray]] = list()
        self.ground_truth = GroundTruth()


    def add_payload(self, payload: Sensor):
        self._payloads.append(payload)


    def advance(self):
        dt = self._sim._time_delta
        self._do_advance(dt)
        self._local_clock += dt


    def post_advance(self):
        for p in self._payloads:
            if p.has_revisited():
                self._sim.meas_queue.append(p.create_measurements())

        self._state_hist.append((self._sim._sim_time, self.state))
        self.ground_truth.update(self.state, self._sim._sim_time)

    
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
