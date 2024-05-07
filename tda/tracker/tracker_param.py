from dataclasses import dataclass
from typing import Callable

from .filters.filter import Filter
from tda.common.measurement import Measurement


@dataclass
class TrackerParam():
    associator_type: str
    initeator_type: str
    deletor_type: str

    filter_factory: Callable[[Measurement], Filter]

    pda_pg: float=0.95
    pda_cr: float=0.15
    pda_init_count: int=4
    pda_initor: str="truth"
    pda_updator: str="truth"