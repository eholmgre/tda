import base64
import json
import os
import pickle

from typing import List

class TrackHist():
    def __init__(self, meas_y, meas_R, meas_t, meas_targ, meas_sensor, state_x, state_P, state_t, state_nis):
        self.meas_y = meas_y
        self.meas_R = meas_R
        self.meas_t = meas_t
        self.meas_targ = meas_targ
        self.meas_sensor = meas_sensor
        self.state_x = state_x
        self.state_P = state_P
        self.state_t = state_t
        self.state_nis = state_nis



def read_tracks(basedir:str ) -> List[TrackHist]:
    track_files = [f for f in os.listdir(basedir) if f.split(".")[-1] == "json"]

    tracks = []

    for t in track_files:
        with open(f"{basedir}/{t}") as f:
            track_dict = json.load(f)

        meas_y = pickle.loads(base64.b64decode(track_dict["meas_y"]))
        meas_R = pickle.loads(base64.b64decode(track_dict["meas_R"]))
        meas_t = pickle.loads(base64.b64decode(track_dict["meas_t"]))
        meas_targ = pickle.loads(base64.b64decode(track_dict["meas_targ"]))
        meas_sensor = pickle.loads(base64.b64decode(track_dict["meas_sensor"]))
        state_x = pickle.loads(base64.b64decode(track_dict["state_x"]))
        state_P = pickle.loads(base64.b64decode(track_dict["state_P"]))
        state_t = pickle.loads(base64.b64decode(track_dict["state_t"]))
        state_nis = pickle.loads(base64.b64decode(track_dict["state_nis"]))

        tracks.append(TrackHist(meas_y, meas_R, meas_t, meas_targ, meas_sensor, state_x, state_P, state_t, state_nis))

    return tracks
