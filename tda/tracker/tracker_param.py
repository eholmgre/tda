import numpy as np
from numpy.typing import NDArray


class TrackerParam():
    def __init__(self, associator_type: str="truth", initeator_type: str="truth", deletor_type: str="time", delete_time: float=60.0,
                 filter_nstate: int=6, filter_startQ: NDArray=np.array([1e4, 1e9, 1e4, 1e9, 1e4, 1e9]),
        filter_n3_q: float=10, filter_n6_q: float=5, filter_n9_q: float=3, filter_turn_q: float=1,
        pda_pg: float=0.95, pda_cr: float=0.15, pda_init_count: int=4, pda_initor: str="truth",
        pda_updator: str="truth", record_tracks: bool=True, record_basename: str="."):
        self.associator_type = associator_type
        self.initeator_type = initeator_type
        self.deletor_type = deletor_type
        self.delete_time = delete_time

        self.filter_nstate = filter_nstate  # 0 - imm, 3, 6, 9
        self.filter_startQ = filter_startQ
        self.filter_n3_q = filter_n3_q
        self.filter_n6_q = filter_n6_q
        self.filter_n9_q = filter_n9_q
        self.filter_turn_q = filter_turn_q

        self.pda_pg = pda_pg
        self.pda_cr = pda_cr
        self.pda_init_count = pda_init_count
        self.pda_initor = pda_initor
        self.pda_updator = pda_updator

        self.record_tracks = record_tracks
        self.record_basename = record_basename