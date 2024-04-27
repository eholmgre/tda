from .sensor import Sensor
from ..sim_objects.sim_object import SimObject

class Radar(Sensor):
    last_meas_time: float
    revisit_rate: float


    def __init__(self, sensor_id: int, host: SimObject, revisit_rate: float, prob_detect: float):
        super().__init__(sensor_id, host, revisit_rate, prob_detect)

