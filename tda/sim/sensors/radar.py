import numpy as np
from numpy.typing import NDArray
import scipy.linalg as la
from scipy.stats import multivariate_normal, uniform
from typing import Any, Dict, List, Optional, Sequence, Tuple

from tda.common.measurement import Measurement

from .sensor import Sensor
from ..sim_objects.sim_object import SimObject

class Radar(Sensor):
    """
    Radar - measure the targets azmuth, elevation and range
    """
    R: NDArray  # measurement cov matrix 3x3 corresponsing to az, el, range meas uncert

    def __init__(self, sensor_id: int, host: SimObject, revisit_rate: float, R: NDArray,
                 prob_detect: float=1.0, field_of_regard: Optional[NDArray]=None):
        super().__init__(sensor_id, host, revisit_rate, prob_detect, field_of_regard)
        self.sensor_type = "radar"

        self.R = R
    
    def _do_create_measurements(self, targets) -> List[Measurement]:
        frame = list()

        for t in targets:
            if t == self._host:
                continue

        
            if self._prob_detect < 1.0:
                if uniform.rvs() > self._prob_detect:
                    continue

            t_pos = t.state[:3]
            my_pos = self._get_sensor_position()

