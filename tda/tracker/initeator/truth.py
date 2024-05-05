from typing import List, Sequence

from tda.common.measurement import Measurement

from .initieator import Initeator
from ..track import Track
from .. import track_bookeeper


class TruthIniteator(Initeator):
    def initeate_tracks(self, frame: Sequence[Measurement]) -> List[Track]:
        new_tracks = list()
        
        for m in frame:
            if m.target_id != 0:
                track_bookeeper.track_id_ctr += 1
                new_tracks.append(Track(track_bookeeper.track_id_ctr, 
                                        track_bookeeper.create_linear_filter(m)))

        return new_tracks
