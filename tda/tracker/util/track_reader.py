import base64
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pymap3d
from scipy.stats import chi2

from typing import List

class TrackHist():
    def __init__(self, trk_id, meas_y, meas_R, meas_t, meas_targ, meas_sensor, state_x, state_P, state_t, state_nis):
        self.track_id = trk_id
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

        name = track_dict["name"]
        meas_y = pickle.loads(base64.b64decode(track_dict["meas_y"]))
        meas_R = pickle.loads(base64.b64decode(track_dict["meas_R"]))
        meas_t = pickle.loads(base64.b64decode(track_dict["meas_t"]))
        meas_targ = pickle.loads(base64.b64decode(track_dict["meas_targ"]))
        meas_sensor = pickle.loads(base64.b64decode(track_dict["meas_sensor"]))
        state_x = pickle.loads(base64.b64decode(track_dict["state_x"]))
        state_P = pickle.loads(base64.b64decode(track_dict["state_P"]))
        state_t = pickle.loads(base64.b64decode(track_dict["state_t"]))
        state_nis = pickle.loads(base64.b64decode(track_dict["state_nis"]))

        tracks.append(TrackHist(name, meas_y, meas_R, meas_t, meas_targ, meas_sensor, state_x, state_P, state_t, state_nis))

    return tracks


def plot_track(track, fig=None, axs=None,):
    state = track.state_x
    t = track.state_t

    tmin = min(t)
    tmax = max(t)

    track_id = track.track_id

    N_state = state.shape[0]
    
    lla = np.zeros((N_state, 3))

    for i in range(N_state):
        x = state[i, 0]
        y = state[i, 3]
        z = state[i, 6]
        lat, lon, alt = pymap3d.ecef2geodetic(x, y, z)
        lla[i] = [lat, lon, alt]

    meas = track.meas_y
    N_meas = meas.shape[0]

    meas_lla = np.zeros((N_meas, 3))

    for i in range(N_meas):
        x = meas[i, 0]
        y = meas[i, 1]
        z = meas[i, 2]
        lat, lon, alt = pymap3d.ecef2geodetic(x, y, z)
        meas_lla[i] = [lat, lon, alt]


    newfig = False
    if fig is None or axs is None:
        fig, axs = plt.subplots(2, 2)
        newfig = True
    axs[0, 0].set_title("lat lon position")
    axs[0, 0].plot(lla[:, 1], lla[:, 0], label=f"track {track_id}")
    axs[0, 1].set_title("altitude over time")
    axs[0, 1].plot(t, lla[:, 2], label=f"track {track_id}")
    axs[1, 0].set_title("chisq over time")
    axs[1, 0].set_ylim([0, 10])
    if newfig:
        chi2_low, chi2_high = chi2.interval(df=3, confidence=0.95)
        axs[1, 0].hlines(y=[chi2_low, chi2_high], xmin=tmin, xmax=tmax, linestyles="dashed", color="grey", label="chi2 95% ci")
    axs[1, 0].plot(t[1:], track.state_nis[1:], label=f"track {track_id}")

    axs[1, 1].set_title("measurements")
    axs[0, 0].scatter(meas_lla[:, 1], meas_lla[:, 0], marker="x", label=f"tracl {track_id}")

    fig.suptitle(f"track {track_id}")
    fig.tight_layout()

    return fig


def plot_tracks(tracks):
    fig, axs = plt.subplots(2, 2)
    
    ids = []

    tmin = tmax = -1
    for t in tracks:
        thistmin = min(t.state_t)
        if tmin < 0 or tmin > thistmin:
            tmin = thistmin
        thistmax = max(t.state_t)
        if tmin < 0 or tmax < thistmax:
            tmax = thistmax

    for t in tracks:
        ids.append(t.track_id)
        plot_track(t, fig=fig, axs=axs)

    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            axs[i, j].legend()

    chi2_low, chi2_high = chi2.interval(df=3, confidence=0.95)
    axs[1, 0].hlines(y=[chi2_low, chi2_high], xmin=tmin, xmax=tmax, linestyles="dashed", color="grey", label="chi2 95% ci")

    idstr = "tracks" + " ".join(map(str, ids))
    fig.suptitle(idstr)

    return fig
