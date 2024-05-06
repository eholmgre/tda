from typing import List, Sequence

from tda.common.measurement import Measurement

from .initieator import Initeator, track_id_ctr
from ..track import Track


class TruthIniteator(Initeator):
    def initeate_tracks(self, frame: Sequence[Measurement]) -> List[Track]:
        global track_id_ctr
        
        new_tracks = list()
        
        for m in frame:
            if m.target_id != 0:
                trk = Track(track_id_ctr, self.create_filter(m))
                track_id_ctr += 1
                trk.update(m)
                new_tracks.append(trk)

        return new_tracks
