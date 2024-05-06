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