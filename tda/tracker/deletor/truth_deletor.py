from typing import List

from .deletor import Deletor
from ..track import Track


class TruthDeletor(Deletor):
    def delete_tracks(self, missed_tracks: List[Track], all_tracks: List[Track], frame_time: float) -> None:
        delete_list = list()
        
        for t in missed_tracks:
            if t.meas_hist[-1].target_id == 0:
                delete_list.append(t)

        for t in delete_list:
            all_tracks.remove(t)