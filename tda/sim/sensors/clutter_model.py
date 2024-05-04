from abc import ABCMeta, abstractmethod
import logging
import numpy as np
from numpy.typing import NDArray
from scipy.stats import expon, poisson, uniform
from typing import List, Union

#from ..sensor import Sensor
from tda.common.measurement import Measurement


class ClutterModel(metaclass=ABCMeta):
    def __init__(self, parent):  #: Sensor):
        self._parent = parent


    @abstractmethod
    def create_clutter_measurements(self) -> List[Measurement]:
        pass


class PoissonClutter(ClutterModel):
    _lambda: float

    def __init__(self, parent, lamb: float):
        super().__init__(parent)
        self._lambda = lamb

    
    def create_clutter_measurements(self) -> List[Measurement]:
        clutter: List[Measurement] = list()

        fov = self._parent._field_of_regard
        dims = fov.shape[0]
        time = self._parent._host._sim._sim_time
        sensor_id = self._parent.sensor_id
        sensor_pos = self._parent._get_sensor_position()

        n = poisson.rvs(self._lambda)

        u = uniform.rvs(size=(n, dims), loc=fov[:, 0], scale=fov[:, 1] - fov[:, 0])

        for i in range(n):
            clutter.append(Measurement(time, sensor_id, 0, "", u[i], sensor_pos))

        return clutter


class PopcornClutter(ClutterModel):
    pass


class TimeCorrelatedClutter(ClutterModel):
    pass
