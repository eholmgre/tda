from typing import List, Sequence

from tda.common.measurement import Measurement

from .initieator import Initeator, track_id_ctr
from ..track import Track


class TruthIniteator(Initeator):
    def initeate_tracks(self, frame: Sequence[Measurement]) -> List[Track]:
        new_tracks = list()
        
        for m in frame:
            if m.target_id != 0:
                track_id_ctr += 1
                new_tracks.append(Track(track_id_ctr,
                self.create_filter(m)))

        return new_tracks
