from abc import ABCMeta, abstractmethod
from typing import Sequence

from tda.common.measurement import Measurement

class Associator(metaclass=ABCMeta):
    @abstractmethod
    def update_scan(self, scan: Sequence[Measurement]):
        pass
