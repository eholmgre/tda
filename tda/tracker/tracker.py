import logging
import sys
from typing import List, Sequence

from .assoc.associator import Associator
from .assoc.pdaf import PDAF
from .assoc.truth import TruthAssociator
from .deletor.deletor import Deletor
from .deletor.miss_based import MissBasedDeletor
from .deletor.truth_deletor import TruthDeletor
from .initeator.initieator import Initeator
from .initeator.truth import TruthIniteator
from .tracker_param import TrackerParam
from .track import Track
from tda.common.measurement import Measurement


class Tracker():
    def __init__(self, params: TrackerParam):
        self.tracks: List[Track] = list()
        self.associator: Associator=None
        self.initeator: Initeator=None
        self.deletor: Deletor=None
        self.params = params

        self.track_id_ctr = 1

        if self.params.associator_type == "truth":
            self.associator = TruthAssociator()
        elif self.params.associator_type == "pda":
            pdaf = PDAF(params.pda_pg, params.pda_cr, params.pda_init_count,
                        params.pda_initor, params.pda_updator,
                        params.filter_factory)
            
            self.associator = pdaf
            self.initeator = pdaf
        else:
            logging.error(f"Unknown associator type: \"{self.params.associator_type}\". Exiting.")
            sys.exit(-1)

        if not self.initeator:
            if self.params.initeator_type == "truth":
                self.initeator = TruthIniteator(self.params.filter_factory)
            else:
                logging.error(f"Unknown initeator type: \"{self.params.initeator_type}\". Exiting.")
                sys.exit(-1)

        if self.params.deletor_type == "truth":
            self.deletor = TruthDeletor()
        elif self.params.deletor_type == "miss_based":
            self.deletor = MissBasedDeletor()
        else:
            logging.error(f"Unknown deletor type: \"{self.params.deletor_type}\". Exiting.")
            sys.exit(-1)


    def process_frame(self, frame: Sequence[Measurement]) -> None:
        if len(frame) == 0:
            return

        frame_time = frame[0].time

        missed_associations, missed_meas = self.associator.assoceate(frame, self.tracks)

        self.tracks.extend(self.initeator.initeate_tracks(missed_meas))

        self.deletor.delete_tracks(missed_associations, self.tracks, frame_time)


    def print_tracks(self):
        print(f" Maintaining: {len(self.tracks)} tracks.")
