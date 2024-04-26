from numpy.typing import NDArray
from scipy.stats import multivariate_normal
from typing import List, Sequence

from .sensor import Sensor
from ..sim_objects.sim_object import SimObject
from tda.common.measurement import Measurement


class OracleMeasurement(Measurement):
    def __init__(self, meas_time: float, sensor_id: int, y: NDArray, sensor_pos: NDArray):
        super().__init__(meas_time, sensor_id, y, sensor_pos)


class Oracle(Sensor):
    """
    Oracle - an all seeing sensor that magically measures the targets position
    """
    R: NDArray  # measurement cov matrix 3x3 corresponsing to x, y, z meas uncert

    def __init__(self, sensor_id: int, host: SimObject, R: NDArray):
        super().__init__(sensor_id, host)

        self.R = R


    def create_measurements(self, targets: Sequence[SimObject]) -> Sequence[Measurement]:
        measList = list()

        for t in targets:
            t_pos = t.state[:3]
            my_pos = self._host.state[:3]
            dispacement = t_pos - my_pos + multivariate_normal.rvs(cov=self.R)

            measList.append(OracleMeasurement(self._host._sim.sim_time,
                                              self.sensor_id,
                                              dispacement,
                                              my_pos))

        return measList
