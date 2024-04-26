from abc import ABCMeta
from typing import Deque
import numpy as np


class SimObject(metaclass=ABCMeta):
    def __init__(self, num_states: int, initial_state: np.array, event_queue: Deque):
        self._num_states = num_states
        self._state = initial_state
        self._local_clock = 0.0
        self._event_queue = event_queue


    def pre_advance():
        pass
    
    def advance(time_quanta: float):
        pass

    def post_advance():
        pass