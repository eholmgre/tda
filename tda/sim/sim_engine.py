from collections import deque
from typing import Any, Deque, Dict, List, Sequence

from .sim_objects.sim_object import SimObject
from tda.common.measurement import Measurement

class Simulation():
    def __init__(self, time_delta: float=0.5, sim_length: float=60):
        self._time_delta = time_delta
        self._sim_length = sim_length
        
        self.records: Dict[str, Dict[str, Any]] = dict()
        self.meas_queue: Deque[Sequence[Measurement]] = deque()
        self._sim_time = 0.0
        self._sim_objects: List[SimObject] = list()


    def setup_sim(self):
        pass


    def run(self) -> Dict[str, Dict[str, Any]]:
        # while we have objects active in the sim and haven't reached the max time
        while self._sim_time <= self._sim_length and len(self._sim_objects):
            # give objects a chance to "set up" for this update
            for s in self._sim_objects:
                s.pre_advance()

            # update the states of the objects to this time step
            for s in self._sim_objects:
                s.advance()

            # let objects act after their update, also check if objects are "done"
            done = list()
            for s in self._sim_objects:
                s.post_advance()

                # we'll clean up in another loop to avoid iterator invalidation
                if s.is_done():
                    done.append(s)

            # remove anything that was "done"
            for s in done:
                for sim_id, sim_hist in s.record():
                    self.records[sim_id] = sim_hist
                
                self._sim_objects.remove(s)

            self._sim_time += self._time_delta

        for s in self._sim_objects:
            for sim_id, sim_hist in s.record():
                self.records[sim_id] = sim_hist
            
        return self.records
