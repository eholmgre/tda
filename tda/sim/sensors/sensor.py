from abc import ABCMeta, abstractmethod
from typing import List, Sequence

from .clutter_models.clutter_model import ClutterModel
from ..sim_objects.sim_object import SimObject
from tda.common.measurement import Measurement


class Sensor(metaclass=ABCMeta):
    sensor_id: int
    _sensor_type: str
    _revisit_rate: float
    _host: SimObject
    _clutter_model: ClutterModel
    _meas_hist: List[Sequence[Measurement]]


    def __init__(self, sensor_id: int, host: SimObject):
        self.sensor_id = sensor_id
        self._host = host


    def create_measurements(self) -> Sequence[Measurement]:
        frame = self._observe_targets()
        self._meas_hist.append(frame)

        return frame


    @abstractmethod
    def _observe_targets(self) -> Sequence[Measurement]:
        pass
