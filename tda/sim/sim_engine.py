
from typing import List

class Simulation():
    def __init__(self, time_quanta: float=0.05):
        self._time_quanta = time_quanta
        self._simobjects = list()