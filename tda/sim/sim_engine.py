from typing import List

from .sim_objects.sim_object import SimObject

class Simulation():
    sim_time: float
    
    _time_quanta: float
    _simobjects: List[SimObject]


    def __init__(self, time_quanta: float=0.05):
        self.sim_time = 0.0
        self._time_quanta = time_quanta
        self._simobjects = list()