from typing import List, Sequence, Tuple

from .associator import Association, Associator
from ..filters.filter import Filter
from ..track import Track
from tda.common.measurement import Measurement


class JDPAF(Associator):
    def assoceate(self, frame: Sequence[Measurement], tracks: Sequence[Track]) -> Tuple[List[Association], List[Track], List[Measurement]]:
        pass