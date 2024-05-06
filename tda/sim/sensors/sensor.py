from abc import ABCMeta, abstractmethod
from numpy.typing import NDArray
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .clutter_model import ClutterModel
from tda.common.measurement import Measurement


class Sensor(metaclass=ABCMeta):
    def __init__(self, sensor_id: int, sensor_type: str, host: "tda.sim.sim_objects.sim_object.SimObject",
                 revisit_rate: float, R: NDArray,
                 prob_detect: float=1.0, field_of_regard: Optional[NDArray]=None,
                 reported_R: Optional[NDArray]=None):
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self._host = host
        self._revisit_rate = revisit_rate
        self.R = R
        if reported_R:
            self.reported_R = reported_R
        else:
            self.reported_R = R
        self._prob_detect = prob_detect
        assert 0.0 <= self._prob_detect <= 1.0
        self._field_of_regard = field_of_regard

        self._last_meas_time = -2 * self._revisit_rate
        self._clutter_model: Optional[ClutterModel] = None

        self._meas_hist: List[List[Measurement]]=list()

    
    def add_clutter_model(self, clutter_model: ClutterModel):
        self._clutter_model = clutter_model

    
    def has_revisited(self):
        return self._host._sim._sim_time - self._last_meas_time >= self._revisit_rate

    
    def check_fov(self, meas: Measurement) -> bool:
        if self._field_of_regard is None:
            return True

        for i in range(meas.y.shape[0]):
            if not (self._field_of_regard[i, 0] <= meas.y[i] <= self._field_of_regard[i, 1]):
                return False

        return True


    def create_measurements(self) -> Sequence[Measurement]:
        targets = self._host._sim._sim_objects
        frame = self._do_create_measurements(targets)
        
        if self._clutter_model:
            frame.extend(self._clutter_model.create_clutter_measurements())
        
        self._meas_hist.append(frame)
        self._last_meas_time = self._host._sim._sim_time

        return frame


    def _create_measurement(self, y: NDArray, target_id: int) -> Measurement:
        return Measurement(self._host._sim._sim_time,
                           self.sensor_id,
                           target_id,
                           self.sensor_type,
                           y,
                           self._get_sensor_position(),
                           self.reported_R,
                           self._prob_detect,
                           self._revisit_rate)
    

    def get_name(self) -> str:
        return f"{self.sensor_type}{self.sensor_id}"


    @abstractmethod
    def _do_create_measurements(self, targets) -> List[Measurement]:
        "sensor specific routine to create the measurements"
        pass


    def _get_sensor_position(self):
        return self._host.state[:3]


    @abstractmethod
    def record(self) -> Tuple[str, Dict[str, Any]]:
        "routing to create the record from this sensor"
        pass
