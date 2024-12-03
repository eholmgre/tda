from pykml import parser
from dateutil.parser import parse
import time
import json
import pickle

import numpy as np
import matplotlib.pyplot as plt
import pymap3d
from scipy.interpolate import interp1d
from scipy.stats import multivariate_normal


def get_interpolator(coordstr):
    times = []
    coords = []

    for line in coordstr.split("\n"):
        if "when" in line:
            datestamp = line[7:-7]
            t = time.mktime(parse(datestamp).timetuple())
            times.append(t)
    
        if "coord" in line:
            cordstr = line[11:-11]
            lon, lat, alt = map(float, cordstr.split())
            coords.append((lat, lon, alt))

    t = np.array(times)
    lla = np.array(coords)

    return interp1d(t, lla, axis=0), min(times), max(times), np.average(np.diff(times))


def get_truth(intrpl, tmin, tmax, dt):
    ts = np.arange(tmin, tmax, dt)
    lla = intrpl(ts)
    return np.hstack((lla, ts.reshape(-1, 1)))


def get_err_meas(intrpl, tmin, tmax, dt, Q):
    ts = np.arange(tmin, tmax, dt)
    lla = intrpl(ts)

    lla_err = np.zeros_like(lla)
    for i in range(lla.shape[0]):
        lat, lon, alt = lla[i]
        x, y, z = pymap3d.geodetic2ecef(lat, lon, alt)
        xerr, yerr, zerr = [x, y, z] + multivariate_normal.rvs(cov=Q)

        lla_err[i] = pymap3d.ecef2geodetic(xerr, yerr, zerr)

    return np.hstack((ts.reshape(-1, 1), lla_err))


def make_measurements(cordstr, hexid, Q=np.eye(3) * (25**2), nac_p=9):
    intrpl, tmin, tmax, dt = get_interpolator(cordstr)
    err_meas = get_err_meas(intrpl, tmin, tmax, dt, Q)
    truth = get_truth(intrpl, tmin, tmax, dt)

    meas_list = []

    for i in range(err_meas.shape[0]):
        t, lat, lon, alt = err_meas[i]

        meas_list.append(
            {
                "now" : t,
                "hex" : hexid,
                "lat" : lat,
                "lon" : lon,
                "alt_geom" : alt,
                "nac_p" : nac_p
            }
        )

    return meas_list, truth


def make_meas_list(meas):
    tot = []
    for m in meas:
        tot.extend(m)

    tot.sort(key=lambda x: x["now"])

    return tot


def write_meas_file(meas, fname):
    with open(fname, "w") as f:
        for m in meas:
            f.write(json.dumps(m) + "\n")


def write_truth_file(truth, fname):
    with open(fname, "wb") as f:
        pickle.dump(truth, f)


def process_cords(cordstrs, hexids, basename, Q=np.eye(3) * (25**2), nac_p=9):
    meas_list = []
    for c, h in zip(cordstrs, hexids):
        meas, truth = make_measurements(c, h, Q, nac_p)
        meas_list.append(meas)

        write_truth_file(truth, f"{basename}/{h}.pkl") 

    write_meas_file(make_meas_list(meas_list), f"{basename}/{''.join([h + '_' for h in hexids])}.json")
