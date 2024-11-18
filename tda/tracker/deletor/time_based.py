from typing import List

from .deletor import Deletor
from ..track import Track


class TimeBasedDeletor(Deletor):
    def __init__(self, delete_time=120):
        self.delete_time = delete_time


    def delete_tracks(self, missed_tracks: List[Track], all_tracks: List[Track], frame_time: float) -> None:
        for t in missed_tracks:
            last_time = t.meas_hist[-1].time
            # if the track only has one hit and we've already missed - it's probably clutter so delete
            if frame_time - last_time >= self.delete_time:
                all_tracks.remove(t)
