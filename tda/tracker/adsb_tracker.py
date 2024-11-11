#! /usr/bin/env python3

from tda.common.measurement import Measurement

from tda.tracker.filters.linear_kalman import LinearKalman
from tda.tracker.track import Track
from tda.tracker.tracker import Tracker
from tda.tracker.tracker_param import TrackerParam

import numpy as np
import pymap3d

from abc import ABCMeta
import argparse
import json
import logging
import socket
import sys




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
    x0_hat = np.array([meas.y[0], 0, 0, meas.y[1], 0, 0, meas.y[2], 0, 0])
    P0_hat = np.eye(9) * 1e9

    return LinearKalman(x0_hat, P0_hat, F, H, Q, np.zeros((3, 3)))


class TargetIDAssigner():
    # singleton - static vars
    _counter = 1
    _target_dict = dict()

    @staticmethod
    def get_id(hex_id):
        if hex_id not in TargetIDAssigner._target_dict:
            TargetIDAssigner._target_dict[hex_id] = TargetIDAssigner._counter
            TargetIDAssigner._counter += 1

        return TargetIDAssigner._target_dict[hex_id]


class ADSBMeasurement(Measurement):
    def __init__(self, protodict):
        meastime = float(protodict["now"])
        self.hex_id = protodict["hex"]
        #self.flight_id = protodict["flight"].strip()
        #self.alt_baro = float(protodict["alt_baro"])
        self.alt_geom = float(protodict["alt_geom"])
        #self.ground_speed = float(protodict["gs"])
        #self.track_head = float(protodict["track"])
        #self.baro_rate = float(protodict["baro_rate"])
        #self.squawk = protodict["squawk"]
        #self.emergency = protodict["emergency"]
        #self.nav_qnh = float(protodict["nav_qnh"])
        #self.nav_altitude_mcp = float(protodict["nav_altitude_mcp"])
        #self.nav_heading = float(protodict["nav_heading"])
        self.lat = float(protodict["lat"])
        self.lon = float(protodict["lon"])
        #self.r_dst = float(protodict["r_dst"])
        #self.r_dir = float(protodict["r_dir"])
        self.nac_p = float(protodict["nac_p"])
        #self.nac_v = float(protodict["nac_v"])

        pos_ecef = pymap3d.geodetic2ecef(self.lat, self.lon, self.alt_geom)
        err_ecef = np.eye(3) * pow(nac2P(self.nac_p), 2)
        target_id = TargetIDAssigner.get_id(self.hex_id)

        super().__init__(meastime, -1, target_id, "adsb", pos_ecef,
                         np.array([0, 0, 0]), err_ecef, 1.0, -1.0)


class InputStream(metaclass=ABCMeta):
    # does the stream have more messages to send
    def have_messages(self) -> bool:
        pass

    def get_next(self) -> ADSBMeasurement:
        pass


class FileInputStream(InputStream):
    def __init__(self, filename):
        self.file = open(filename)
        self.has_data = True


    def __del__(self):
        self.file.close()


    def have_messages(self) -> bool:
        return self.has_data
    

    def get_next(self) -> ADSBMeasurement:
        line = self.file.readline()
        
        # hacky? this was the last line
        if len(line) == 0 or line[-1] != "\n":
            self.has_data = False
            return None

        try:
            j = json.loads(line)
        except json.JSONDecodeError as e:
            logging.warning(f"Invalid json line: {line}")

        for k in ("now", "lat", "lon", "alt_geom", "nac_p", "hex"):
            if k not in j:
                logging.warning(f"discarding bad message, no {k}")
                return None
            
        return ADSBMeasurement(j)
    

class JsonTCPStream(InputStream):
    def __init__(self, host, port):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect((host, port))


    def have_messages(self) -> bool:
        return True
    

    def get_next(self) -> ADSBMeasurement:
        data = self.client.recv(2048)

        try:
            j = json.loads(data)
        except json.JSONDecodeError as e:
            logging.warning(f"Bad Json string received : {data}")

        return ADSBMeasurement(j)


def get_opts():
    parser = argparse.ArgumentParser(
        prog="ADSB Tracker",
        description="Ingests Mode-S messages and forms state vectors",
    )

    parser.add_argument("-n", "--numprosc", help="Number of processes to spawn. NOT SUPPORTED YET.")

    ingroup = parser.add_mutually_exclusive_group(required=True)
    ingroup.add_argument("-f", "--filename", help="parse json encoded file as fast as possible.")
    ingroup.add_argument("-j", "--jsontcp", help="connects to json encoded tcp stream from dump1090. Format is host:<port> default port is 30154.")
    ingroup.add_argument("-r", "--rawtcp", help="connects to raw mode-s stream from dump1090. Format is host:<port> where default port is 30005. NOT SUPPORTED YET.")

    return parser.parse_args()


def main(opts):

    tracker_params = TrackerParam(associator_type="truth",
                            initeator_type="truth",
                            deletor_type="truth",
                            filter_factory=lkf_factory)

    tracker = Tracker(tracker_params)

    if opts.filename:
        input_stream = FileInputStream(opts.filename)
    elif opts.jsontcp:
        s = opts.jsontcp.split(":")

        host = s[0]
        if len(s) == 1:
            port = 30154
        elif len(2) == 2:
            port = int(s[1])
        else:
            logging.error(f"Invalid json tcp stream: {opts.jsontcp}")
            sys.exit(1)

        input_stream = JsonTCPStream(host, port)

    else:
        logging.error("Invalid data input.")
        sys.exit(1)

    while input_stream.have_messages():
        message = input_stream.get_next()

        if not message:
            continue

        tracker.process_frame([message])
        tracker.print_tracks()


if __name__ == "__main__":
    opts = get_opts()
    main(opts)