from numpy.typing import NDArray


class Measurement():
    time: float  # time of measurement creation
    sensor_id: int  # id of sensor which created the measurement
    target_id: int  # id of target which emited the measurement (0 for clutter)
    measurement_type: str  # type of measurement aka radar, los, ...
    y: NDArray  # measurement data
    sensor_pos: NDArray


    def __init__(self, meas_time: float, sensor_id: int, target_id: int, meas_type: str,
                 y: NDArray, sensor_pos: NDArray):
        self.time = meas_time
        self.sensor_id = sensor_id
        self.target_id = target_id
        self.measurement_type = meas_type
        self.y = y
        self.sensor_pos = sensor_pos
