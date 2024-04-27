from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .clutter_models.clutter_model import ClutterModel
# from ..sim_objects.sim_object import SimObject
from tda.common.measurement import Measurement


class Sensor(metaclass=ABCMeta):
    sensor_id: int
    sensor_type: str
    _revisit_rate: float
    _last_meas_time: float
    #_host: SimObject
    _clutter_model: Optional[ClutterModel]
    _meas_hist: List[Sequence[Measurement]]


    def __init__(self, sensor_id: int, host, revisit_rate: float): #: SimObject):
        self.sensor_id = sensor_id
        self._host = host
        self._revisit_rate = revisit_rate

        self._last_meas_time = -2 * self._revisit_rate
        self._clutter_model = None
        self._meas_hist = list()

    
    def add_clutter_model(self, clutter_model: ClutterModel):
        self._clutter_model = clutter_model

    
    def has_revisited(self):
        return self._host._sim._sim_time - self._last_meas_time >= self._revisit_rate


    def create_measurements(self) -> Sequence[Measurement]:
        targets = self._host._sim._sim_objects
        frame = self._do_create_measurements(targets)
        if self._clutter_model:
            frame.extend(self._clutter_model.create_clutter_measurements())
        
        self._meas_hist.append(frame)
        self._last_meas_time = self._host._sim._sim_time

        return frame
    

    def get_name(self) -> str:
        return f"{self.sensor_type}{self.sensor_id}"


    @abstractmethod
    def _do_create_measurements(self, targets) -> List[Measurement]:
        "sensor specific routine to create the measurements"
        pass


    @abstractmethod
    def record(self) -> Tuple[str, Dict[str, Any]]:
        "routing to create the record from this sensor"
        pass