import logging
import sys
from typing import List, Sequence

from .assoc.associator import Associator
from .assoc.truth import TruthAssociator
from .deletor.deletor import Deletor
from .deletor.miss_based import MissBasedDeletor
from .initeator.initieator import Initeator
from .initeator.truth import TruthIniteator
from.track import Track
from tda.common.measurement import Measurement


class Tracker():
    def __init__(self, associator_type: str, initeator_type: str, deletor_type: str):
        self.tracks: List[Track] = list()
        self.associator: Associator
        self.initeator: Initeator
        self.deletor: Deletor

        if associator_type == "truth":
            self.associator = TruthAssociator()
        else:
            logging.error(f"Unknown associator type: \"{associator_type}\". Exiting.")
            sys.exit(-1)

        if initeator_type == "truth":
            self.initeator = TruthIniteator()
        else:
            logging.error(f"Unknown initeator type: \"{initeator_type}\". Exiting.")
            sys.exit(-1)

        if deletor_type == "miss_based":
            self.deletor = MissBasedDeletor()
        else:
            logging.error(f"Unknown deletor type: \"{deletor_type}\". Exiting.")
            sys.exit(-1)



    def process_frame(self, frame: Sequence[Measurement]) -> None:
        associations, missed_associations, missed_meas = self.associator.assoceate(frame, self.tracks)

        for a in associations:
            a.track.update(a.meas)

        self.tracks.extend(self.initeator.initeate_tracks(missed_meas))

        self.deletor.delete_tracks(missed_associations, self.tracks, frame[0].time)