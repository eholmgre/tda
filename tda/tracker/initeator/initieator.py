from abc import ABCMeta, abstractmethod
from typing import List, Sequence

from ..track import Track
from tda.common.measurement import Measurement


class Initeator(metaclass=ABCMeta):
    @abstractmethod
    def initeate_tracks(self, frame: Sequence[Measurement]) -> List[Track]:
        pass
