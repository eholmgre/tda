from typing import List, Sequence, Tuple

from .associator import Associator, Association
from ..track import Track
from tda.common.measurement import Measurement


class TruthAssociator(Associator):
    def assoceate(self, frame: Sequence[Measurement], tracks: Sequence[Track]) \
        -> Tuple[List[Track], List[Measurement]]:

        associations = list()
        found_meas = list()

        unassigned_tracks = list()
        unassigned_meas = list()

        for t in tracks:
            found = False
            for m in frame:
                if m.target_id == t.meas_hist[-1].target_id:
                    associations.append(Association(t, m, 1.0))
                    found = True
                    found_meas.append(m)

            if not found:
                unassigned_tracks.append(t)

        for m in frame:
            if m not in found_meas:
                unassigned_meas.append(m)

        for a in associations:
            a.track.update(a.meas)

        return unassigned_tracks, unassigned_meas
