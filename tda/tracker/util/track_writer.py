import base64
import json
import numpy as np
import pickle
from typing import Dict, Union

from tda.common.measurement import Measurement
from tda.tracker.track import Track

class TrackWriter():
    def __init__(self, do_write, basename):
        self._do_write = do_write
        self._basename = basename


    def write_track(self, track: Track) -> None:
        if not self._do_write:
            return
        
        hist_dict : Dict[str, Union[int, str]] = {
            "name" : track.track_id,
        }

        N_meas = len(track.meas_hist)
        P_meas = 3 # track.meas_hist[0].y.shape[0]
        meas_y = np.zeros((N_meas, P_meas))
        meas_R = np.zeros((N_meas, P_meas, P_meas))
        meas_t = np.zeros(N_meas)
        meas_targ = np.zeros(N_meas)
        meas_sensor = np.zeros(N_meas)

        for i, m in enumerate(track.meas_hist):
            meas_y[i] = m.y
            meas_R[i] = m.sensor_cov
            meas_t[i] = m.time
            meas_targ[i] = m.target_id
            meas_sensor[i] = m.sensor_id

        hist_dict["meas_y"] = base64.b64encode(pickle.dumps(meas_y)).decode()
        hist_dict["meas_R"] = base64.b64encode(pickle.dumps(meas_R)).decode()
        hist_dict["meas_t"] = base64.b64encode(pickle.dumps(meas_t)).decode()
        hist_dict["meas_targ"] = base64.b64encode(pickle.dumps(meas_targ)).decode()
        hist_dict["meas_sensor"] = base64.b64encode(pickle.dumps(meas_sensor)).decode()

        hist_dict["state_x"] = base64.b64encode(pickle.dumps(np.array(track.state_hist.state))).decode()
        hist_dict["state_P"] = base64.b64encode(pickle.dumps(np.array(track.state_hist.cov))).decode()
        hist_dict["state_t"] = base64.b64encode(pickle.dumps(np.array(track.state_hist.time))).decode()
        hist_dict["state_score"] = base64.b64encode(pickle.dumps(np.array(track.state_hist.score))).decode()

        hist_dict["state_pos"] = base64.b64encode(pickle.dumps(np.array(track.state_hist.pos))).decode()
        hist_dict["state_sig_pos"] = base64.b64encode(pickle.dumps(np.array(track.state_hist.sig_pos))).decode()
        hist_dict["state_vel"] = base64.b64encode(pickle.dumps(np.array(track.state_hist.vel))).decode()
        hist_dict["state_sig_vel"] = base64.b64encode(pickle.dumps(np.array(track.state_hist.sig_vel))).decode()
        hist_dict["state_accel"] = base64.b64encode(pickle.dumps(np.array(track.state_hist.accel))).decode()
        hist_dict["state_sig_accel"] = base64.b64encode(pickle.dumps(np.array(track.state_hist.sig_accel))).decode()
        
        with open(f"{self._basename}/{track.track_id}.json", "w") as f:
            json.dump(hist_dict, f)