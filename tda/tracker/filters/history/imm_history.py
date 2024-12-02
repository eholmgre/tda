import base64
import numpy as np
from numpy.typing import NDArray
import pickle
from typing import Dict, List

#from ..imm import IMM
from .filter_history import FilterHistory
from .linear_kalman_history import LinearKalmanHistory


class LinearKalmanManauverHistory(LinearKalmanHistory):
    def __init__(self, filt: "tda.filter.imm.LinearKalmanManauver"):
        super().__init__(filt)
        self.filt: "tda.filter.imm.LinearKalmanManauver"
        self.omega: List[float] = []

    def record(self) -> None:
        super().record()

        self.omega.append(self.filt.update_omega)

    
    def save(self) -> Dict:
        base_dict = super().save()

        base_dict["ma_omega"] = base64.b64encode(pickle.dumps(np.array(self.omega))).decode()

        return base_dict
    

    def read(self, hist_dict) -> None:
        super().read(hist_dict)

        self.omega = pickle.loads(base64.b64decode(hist_dict["ma_omega"]))


class IMMHistory(FilterHistory):
    def __init__(self, filt: "tda.filter.IMM"):
        super().__init__(filt, "imm")
        self.filt: "tda.filter.IMM"
        self.mu: List[NDArray] = []


    def record(self) -> None:
        super().record()

        self.mu.append(self.filt.mu)


    def save(self) -> Dict:
        base_dict = super().save()

        base_dict["imm_mu"] = base64.b64encode(pickle.dumps(np.array(self.mu))).decode()

        cv_dict = self.filt.cv_filter.record()
        ca_dict = self.filt.ca_filter.record()
        ma_dict = self.filt.manuver_filter.record()

        for k, v in cv_dict.items():
            base_dict[f"cv_{k}"] = v

        for k, v in ca_dict.items():
            base_dict[f"ca_{k}"] = v

        for k, v in ma_dict.items():
            base_dict[f"ma_{k}"] = v

        return base_dict


    def read(self, hist_dict):
        super().read(hist_dict)
        self.mu = pickle.loads(base64.b64decode(hist_dict["imm_mu"]))
