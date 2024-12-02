import base64
import json
import numpy as np
import pickle
from typing import Dict, Union

from tda.common.measurement import Measurement
from tda.tracker.track import Track
from tda.tracker.conflict import Conflict


class TrackWriter():
    def __init__(self, do_write, basename):
        self._do_write = do_write
        self._basename = basename


    def write_track(self, track: Track) -> None:
        if not self._do_write:
            return
        
        hist_dict = track.filter.record()
        
        hist_dict["name"] = track.track_id

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
        
        with open(f"{self._basename}/{track.track_id}.json", "w") as f:
            json.dump(hist_dict, f)


    def write_conflict(self, conflict: Conflict) -> None:
        conf_dict = conflict.save()
        conf_dict["filter_type"] = "conflict"

        name = conf_dict["name"]
        
        with open(f"{self._basename}/{name}.json", "w") as f:
            json.dump(conf_dict, f)