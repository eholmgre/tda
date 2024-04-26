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


    def __init__(self, sensor_id: int, host: SimObject):
        self.sensor_id = sensor_id
        self._host = host

    @abstractmethod
    def create_measurements(self, targets: Sequence[SimObject]) -> List[Measurement]:
        pass
