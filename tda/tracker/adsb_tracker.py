#! /usr/bin/env python3

from tda.common.measurement import Measurement

from tda.tracker.filters.linear_kalman import LinearKalman
from tda.tracker.track import Track
from tda.tracker.tracker import Tracker
from tda.tracker.tracker_param import TrackerParam

import numpy as np
import pymap3d

import json
import socket


def nac2P(nac):
    # https://mode-s.org/decode/content/ads-b/7-uncertainty.html
    sigma = 37040
    if nac == 9:
        sigma = 7.5
    elif nac == 8:
        sigma = 25
    elif nac == 7:
        sigma = 185
    elif nac == 6:
        sigma = 370
    elif nac == 5:
        sigma = 926
    elif nac == 4:
        sigma = 1852
    elif nac == 3:
        sigma = 3704
    elif nac == 2:
        sigma = 18520

    return np.eye(3) * sigma ** 2

def F(dt: float):
    F = np.eye(9, dtype=np.float64)
    F[0, 1] = F[1, 2] = F[3, 4] = F[4, 5] = F[6, 7] = F[7, 8] = dt
    F[0, 2] = F[3, 5] = F[6, 8] = (dt ** 2) / 2

    return F

def Q(dt: float):
    Q = np.zeros((9, 9))
    Q[0, 0] = 0.35
    Q[1, 1] = 0.35
    Q[2, 2] = 0.35

    return dt * Q

H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0, 0]])

def hinv(y):
    return np.array([y[0], y[1], y[2], 0, 0, 0])

def lkf_factory(meas: Measurement) -> Track:
    x0_hat = meas.y
    P0_hat = np.eye(6) * 1e9

    return LinearKalman(x0_hat, P0_hat, F, H, Q, np.zeros((3, 3)))


def main():
    host = "192.168.1.110"
    port = 30154

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((host, port))

    tracker_params = TrackerParam(associator_type="truth",
                              initeator_type="truth",
                              deletor_type="miss_based",
                              filter_factory=lkf_factory)

    tracker = Tracker(tracker_params)

    targets = ["clut"]

    while True:
        data = client.recv(2048)

        try:
            msg = json.loads(data)

            lat = float(msg["lat"])
            lon = float(msg["lon"])
            alt = float(msg["alt_geom"])
            time = float(msg["now"])
            nac_p = int(msg["nac_p"])
            nac_v = msg["nac_v"]
            flight = msg["flight"].strip()

            x, y, z = pymap3d.geodetic2ecef(lat, lon, alt)

            if flight not in targets:
                targets.append(flight)

            meas = Measurement(time, -1, targets.index(flight), "adsb", np.array([x, y, z]), np.zeros(3), nac2P(nac_p), 0.99, 0.0)

            tracker.process_frame([meas])
            tracker.print_tracks()

            # print(f"{flight} ({time}): lat: {lat}, lon: {lon}, alt: {alt}, ecef x: {x}, y: {y}, z: {z} (nac: {nac_p}, {nac_v})")
        except KeyError as e:
            print(e + " skipping.")

if __name__ == "__main__":
    main()