from typing import List, Sequence, Tuple

from tda.common.measurement import Measurement
from tda.tracker.track import Track

from .associator import Associator, Association


class TruthAssociator(Associator):
    def assoceate(self, frame: Sequence[Measurement], tracks: Sequence[Track]) \
        -> Tuple[List[Association], List[Track], List[Measurement]]:

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

        return associations, unassigned_tracks, unassigned_meas
