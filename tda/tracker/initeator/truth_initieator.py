from typing import List, Sequence

from .initieator import Initeator, track_id_ctr
from ..track import Track
from tda.common.measurement import Measurement


class TruthIniteator(Initeator):
    def initeate_tracks(self, frame: Sequence[Measurement]) -> List[Track]:
        global track_id_ctr
        
        new_tracks: List[Track] = list()
        
        for m in frame:
            if m.target_id != 0:
                trk = Track(track_id_ctr, self.create_filter(m), m)
                track_id_ctr += 1
                #trk.update(m)
                new_tracks.append(trk)

        return new_tracks
