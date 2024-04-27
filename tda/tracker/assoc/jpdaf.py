from typing import List, Sequence

from .associator import Associator
from ..filters.filter import Filter
from tda.common.measurement import Measurement


class JDPAF(Associator):
    filters: List[Filter]


    def __init__(self):
        filters = list()

    
    def add_filter(self, filter: Filter):
        self.filters.append(filter)
    

    def update_scan(self, scan: Sequence[Measurement]):
        pass
