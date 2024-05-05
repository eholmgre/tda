from abc import ABCMeta, abstractmethod
from typing import List

from ..track import Track


class Deletor(metaclass=ABCMeta):
    @abstractmethod
    def delete_tracks(self, missed_tracks: List[Track], all_tracks: List[Track], frame_time: float) -> None:
        pass
