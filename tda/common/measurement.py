from abc import ABCMeta
from numpy.typing import NDArray


class Measurement(metaclass=ABCMeta):
    time: float  # time of measurement creation
    sensor_id: int  # id of sensor which created the measurement
    measurement_type: str  # type of measurement aka radar, los, ...
    y: NDArray  # measurement data
    sensor_pos: NDArray


    def __init__(self, meas_time: float, sensor_id: int, y: NDArray, sensor_pos: NDArray):
        self.time = meas_time
        self.sensor_id = sensor_id
        self.y = y
        self.sensor_pos = sensor_pos
