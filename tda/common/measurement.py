import numpy as np
from numpy.typing import NDArray


class Measurement():
    def __init__(self, meas_time: float, sensor_id: int, target_id: int, meas_type: str,
                  y: NDArray, sensor_pos: NDArray, sensor_cov: NDArray, sensor_pd: float,
                  sensor_revisit: float):
        self.time = meas_time
        self.sensor_id = sensor_id
        self.target_id = target_id
        self.measurement_type = meas_type
        self.y = y
        self.sensor_pos = sensor_pos
        self.sensor_cov = sensor_cov
        self.sensor_pd = sensor_pd
        self.sensor_revisit = sensor_revisit
