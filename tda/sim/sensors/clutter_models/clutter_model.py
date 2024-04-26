from abc import ABCMeta, abstractmethod
from numpy.typing import NDArray
from typing import List

from tda.common.measurement import Measurement


class ClutterModel(metaclass=ABCMeta):
    _extent: NDArray


    def __init__(self, extent: NDArray):
        self._extent = extent


    @abstractmethod
    def create_clutter_measurements(self) -> List[Measurement]:
        pass


class PopcornClutter(ClutterModel):
    pass


class PoissonClutter(ClutterModel):
    pass
