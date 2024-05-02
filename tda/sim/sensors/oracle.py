import numpy as np
from numpy.typing import NDArray
from scipy.stats import multivariate_normal, uniform
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .sensor import Sensor
from ..sim_objects.sim_object import SimObject
from tda.common.measurement import Measurement


class Oracle(Sensor):
    """
    Oracle - an all seeing sensor that magically measures the targets position
    """
    R: NDArray  # measurement cov matrix 3x3 corresponsing to x, y, z meas uncert

    def __init__(self, sensor_id: int, host: SimObject, revisit_rate: float, R: NDArray,
                 prob_detect: float=1.0, field_of_regard: Optional[NDArray]=None):
        super().__init__(sensor_id, host, revisit_rate, prob_detect, field_of_regard)
        self.sensor_type = "oracle"

        self.R = R


    def _do_create_measurements(self, targets: Sequence[SimObject]) -> List[Measurement]:
        frame = list()

        for t in targets:
            # don't measure own platform
            if t == self._host:
                continue

            if self._prob_detect < 1.0:
                if uniform.rvs() > self._prob_detect:
                    continue

            t_pos = t.state[:3]
            my_pos = self._get_sensor_position()
            dispacement = t_pos - my_pos + multivariate_normal.rvs(cov=self.R)

            meas = Measurement(self._host._sim._sim_time,
                               self.sensor_id,
                               t.object_id,
                               "oracle",
                               dispacement,
                               my_pos)

            if self.check_fov(meas):
                frame.append(meas)
            
        return frame
    

    def record(self) -> Tuple[str, Dict[str, Any]]:
        r = dict()

        n = sum([len(f) for f in self._meas_hist])

        r["t"] = np.zeros(n)
        r["sensor_id"] = np.zeros(n)
        r["target_id"] = np.zeros(n)
        r["target_x"] = np.zeros(n)
        r["target_y"] = np.zeros(n)
        r["target_z"] = np.zeros(n)

        r["sensor_x"] = np.zeros(n)
        r["sensor_y"] = np.zeros(n)
        r["sensor_z"] = np.zeros(n)

        i = 0
        for frame in self._meas_hist:
            for meas in frame:
                r["t"][i] = meas.time
                r["sensor_id"][i] = meas.sensor_id
                r["target_id"][i] = meas.target_id

                r["target_x"][i] = meas.y[0]
                r["target_y"][i] = meas.y[1]
                r["target_z"][i] = meas.y[2]

                r["sensor_x"][i] = meas.sensor_pos[0]
                r["sensor_y"][i] = meas.sensor_pos[1]
                r["sensor_z"][i] = meas.sensor_pos[2]
            
                i += 1


        return (self.get_name(), r)