from collections import deque
from typing import Deque, List, Sequence

from .sim_objects.sim_object import SimObject
from tda.common.measurement import Measurement

class Simulation():
    sim_time: float  # current simulation time
    meas_queue: Deque[Sequence[Measurement]]  # queue to hold meas (not sure if I'll use)
    
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


    def run(self):
        while self.sim_time <= self._sim_length:
            # give objects a chance to "set up" for this update
            for s in self._simobjects:
                s.pre_advance()

            # update the states of the objects to this time step
            for s in self._simobjects:
                s.advance(self._time_delta)

            # let objects act after their update
            for s in self._simobjects:
                s.post_advance()
            
