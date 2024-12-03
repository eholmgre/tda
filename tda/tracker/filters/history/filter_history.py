import base64
import numpy as np
from numpy.typing import NDArray
import pickle
from typing import Dict, List, Union

# from ..filter import Filter


class FilterHistory():
    def __init__(self, filt: "tda.filter.Filter", filter_type: str):
        self.filt = filt
        self.filter_type = filter_type
        self.state: List[NDArray] = []
        self.cov: List[NDArray] = []
        self.pos: List[NDArray] = []
        self.sig_pos: List[NDArray] = []
        self.vel: List[NDArray] = []
        self.sig_vel: List[NDArray] = []
        self.accel: List[NDArray] = []
        self.sig_accel: List[NDArray] = []
        self.score: List[float] = []
        self.time: List[float] = []

        self.num_updates: int = 0


    def record(self) -> None:
        self.state.append(self.filt.x_hat)
        self.cov.append(self.filt.P)

        pos, sig_pos = self.filt.get_position()
        self.pos.append(pos)
        self.sig_pos.append(sig_pos)

        vel, sig_vel = self.filt.get_velocity()
        self.vel.append(vel)
        self.sig_vel.append(sig_vel)

        accel, sig_accel = self.filt.get_acceleration()
        self.accel.append(accel)
        self.sig_accel.append(sig_accel)

        self.score.append(self.filt.update_score)
        self.time.append(self.filt.update_time)

        self.num_updates += 1


    def save(self) -> Dict:
        hist_dict : Dict[str, Union[int, str]] = dict()

        hist_dict["filter_type"] = self.filter_type
        hist_dict["num_hits"] = self.num_updates
        hist_dict["state_x"] = base64.b64encode(pickle.dumps(np.array(self.state))).decode()
        hist_dict["state_P"] = base64.b64encode(pickle.dumps(np.array(self.cov))).decode()
        hist_dict["state_t"] = base64.b64encode(pickle.dumps(np.array(self.time))).decode()
        hist_dict["state_score"] = base64.b64encode(pickle.dumps(np.array(self.score))).decode()
        hist_dict["state_pos"] = base64.b64encode(pickle.dumps(np.array(self.pos))).decode()
        hist_dict["state_sig_pos"] = base64.b64encode(pickle.dumps(np.array(self.sig_pos))).decode()
        hist_dict["state_vel"] = base64.b64encode(pickle.dumps(np.array(self.vel))).decode()
        hist_dict["state_sig_vel"] = base64.b64encode(pickle.dumps(np.array(self.sig_vel))).decode()
        hist_dict["state_accel"] = base64.b64encode(pickle.dumps(np.array(self.accel))).decode()
        hist_dict["state_sig_accel"] = base64.b64encode(pickle.dumps(np.array(self.sig_accel))).decode()

        return hist_dict


    def read(self, hist_dict):
        self.state = pickle.loads(base64.b64decode(hist_dict["state_x"]))
        self.cov = pickle.loads(base64.b64decode(hist_dict["state_P"]))
        self.time = pickle.loads(base64.b64decode(hist_dict["state_t"]))
        self.score = pickle.loads(base64.b64decode(hist_dict["state_score"]))
        self.pos = pickle.loads(base64.b64decode(hist_dict["state_pos"]))
        self.state_sig_pos = pickle.loads(base64.b64decode(hist_dict["state_sig_pos"]))
        self.vel = pickle.loads(base64.b64decode(hist_dict["state_vel"]))
        self.sig_vel = pickle.loads(base64.b64decode(hist_dict["state_sig_vel"]))
        self.accel = pickle.loads(base64.b64decode(hist_dict["state_accel"]))
        self.sig_accel = pickle.loads(base64.b64decode(hist_dict["state_sig_accel"]))
