#from ..linear_kalman import LinearKalman3
from .filter_history import FilterHistory


class LinearKalmanHistory(FilterHistory):
    def __init__(self, filt: "tda.filter.LinearKalman3"):
        super().__init__(filt, "linear_kalman")

        self.filt: "tda.filter.LinearKalman3"
