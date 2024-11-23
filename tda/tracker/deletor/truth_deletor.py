from typing import List

from .deletor import Deletor
from ..track import Track
from ..util.track_writer import TrackWriter


class TruthDeletor(Deletor):
    def delete_tracks(self, missed_tracks: List[Track], all_tracks: List[Track], frame_time: float, recorder:TrackWriter=None) -> None:
        delete_list = list()
        
        for t in missed_tracks:
            if t.meas_hist[-1].target_id == 0:
                delete_list.append(t)

        for t in delete_list:
            if recorder:
                recorder.write_track(t)
            all_tracks.remove(t)