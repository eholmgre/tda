from collections import deque
from typing import Any, Deque, Dict, List, Sequence

from .sim_objects.sim_object import SimObject
from tda.common.measurement import Measurement

class Simulation():
    sim_time: float  # current simulation time
    meas_queue: Deque[Sequence[Measurement]]  # queue to hold meas (not sure if I'll use)
    records: Dict[str, Dict[str, Any]]
    
    _time_delta: float  # time step to update targets
    _sim_length: float  # length of time to run engagement
    _simobjects: List[SimObject]  # objects which will act during the engagement


    def __init__(self, time_delta: float=0.5, sim_length: float=60):
        self.meas_queue = deque()

        self._sim_time = 0.0
        self._time_delta = time_delta
        self._sim_length = sim_length
        self._simobjects = list()


    def setup_sim(self):
        pass


    def run(self) -> Dict[str, Dict[str, Any]]:
        # while we have objects active in the sim and haven't reached the max time
        while self.sim_time <= self._sim_length and len(self._simobjects):
            # give objects a chance to "set up" for this update
            for s in self._simobjects:
                s.pre_advance()

            # update the states of the objects to this time step
            for s in self._simobjects:
                s.advance()

            # let objects act after their update, also check if objects are "done"
            done = list()
            for s in self._simobjects:
                s.post_advance()

                # we'll clean up in another loop to avoid iterator invalidation
                if s.is_done():
                    done.append(s)

            # remove anything that was "done"
            for s in done:
                for sim_id, sim_hist in s.record():
                    self.records[sim_id] = sim_hist
                
                self._simobjects.remove(s)

        for s in self._simobjects:
            for sim_id, sim_hist in s.record():
                self.records[sim_id] = sim_hist
            
        return self.records
