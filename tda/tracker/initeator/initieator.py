from abc import ABCMeta, abstractmethod
from typing import Callable, List, Sequence

from ..filters.filter import Filter
from ..track import Track
from tda.common.measurement import Measurement


track_id_ctr = 0


class Initeator(metaclass=ABCMeta):
    def __init__(self, create_filter: Callable[[Measurement], Filter]) -> None:
        self.create_filter = create_filter

    @abstractmethod
    def initeate_tracks(self, frame: Sequence[Measurement]) -> List[Track]:
        pass
