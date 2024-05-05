from abc import ABCMeta, abstractmethod
from typing import List, Sequence, Tuple

from ..track import Track
from tda.common.measurement import Measurement


class Association():
    track: Track
    meas: Measurement
    score: float

    def __init__(self, track: Track, meas: Measurement, score: float) -> None:
        self.track = track
        self.meas = meas
        self.score = score


class Associator(metaclass=ABCMeta):
    @abstractmethod
    def assoceate(self, frame: Sequence[Measurement], tracks: Sequence[Track]) \
        -> Tuple[List[Association], List[Track], List[Measurement]]:
        pass
