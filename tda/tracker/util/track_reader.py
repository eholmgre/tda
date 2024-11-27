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
    def __init__(self, trk_id, meas_y, meas_R, meas_t, meas_targ, meas_sensor,
                 state_x, state_P, state_t, state_score, state_pos, state_sig_pos,
                 state_vel, state_sig_vel, state_accel, state_sig_accel):
        self.track_id = trk_id
        self.meas_y = meas_y
        self.meas_R = meas_R
        self.meas_t = meas_t
        self.meas_targ = meas_targ
        self.meas_sensor = meas_sensor
        self.state_x = state_x
        self.state_P = state_P
        self.state_t = state_t
        self.state_score = state_score

        self.state_pos = state_pos
        self.state_sig_pos = state_sig_pos

        self.state_vel = state_vel
        self.state_sig_vel = state_sig_vel

        self.state_accel = state_accel
        self.state_sig_accel = state_sig_accel


def read_tracks(basedir:str ) -> List[TrackHist]:
    track_files = [f for f in os.listdir(basedir) if f.split(".")[-1] == "json"]

    tracks: List[TrackHist] = []

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
        state_score = pickle.loads(base64.b64decode(track_dict["state_score"]))

        state_pos = pickle.loads(base64.b64decode(track_dict["state_pos"]))
        state_sig_pos = pickle.loads(base64.b64decode(track_dict["state_sig_pos"]))
        state_vel = pickle.loads(base64.b64decode(track_dict["state_vel"]))
        state_sig_vel = pickle.loads(base64.b64decode(track_dict["state_sig_vel"]))
        state_accel = pickle.loads(base64.b64decode(track_dict["state_accel"]))
        state_sig_accel = pickle.loads(base64.b64decode(track_dict["state_sig_accel"]))

        tracks.append(TrackHist(name, meas_y, meas_R, meas_t, meas_targ, meas_sensor,
                                state_x, state_P, state_t, state_score,
                                state_pos, state_sig_pos, state_vel, state_sig_vel,
                                state_accel, state_sig_accel))

    return tracks


def plot_track(track, fig=None, axs=None,):
    state_pos = track.state_pos
    state_t = track.state_t

    tmin = min(state_t)
    tmax = max(state_t)

    track_id = track.track_id

    N_state = state_pos.shape[0]
    
    lla = np.zeros((N_state, 3))

    for i in range(N_state):
        x = state_pos[i, 0]
        y = state_pos[i, 1]
        z = state_pos[i, 2]
        lat, lon, alt = pymap3d.ecef2geodetic(x, y, z)
        lla[i] = [lat, lon, alt]

    meas = track.meas_y
    N_meas = meas.shape[0]
    meas_t = track.meas_t

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
    axs[0, 0].scatter(meas_lla[:, 1], meas_lla[:, 0], alpha=0.5, marker="x", label=f"meas {track_id}")
    axs[0, 1].set_title("altitude over time")
    axs[0, 1].plot(state_t, lla[:, 2], label=f"track {track_id}")
    axs[0, 1].scatter(meas_t, meas_lla[:, 2], alpha=0.5, marker="x", label=f"meas {track_id}")
    axs[1, 0].set_title("chisq over time")
    axs[1, 0].set_ylim([0, 10])
    if newfig:
        chi2_low, chi2_high = chi2.interval(df=3, confidence=0.95)
        axs[1, 0].hlines(y=[chi2_low, chi2_high], xmin=tmin, xmax=tmax, linestyles="dashed", color="grey", label="chi2 95% ci")
    axs[1, 0].plot(state_t[1:], track.state_score[1:], label=f"track {track_id}")

    axs[1, 1].set_title("meas dt")
    axs[1, 1].plot(np.diff(meas_t))
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