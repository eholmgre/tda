import base64
import numpy as np
from numpy.typing import NDArray
import pickle
import pymap3d
import scipy.linalg as la
from scipy.stats import multivariate_normal as mvn
from typing import Dict, List, Tuple, Union

from tda.tracker.track import Track


class Conflict():
    def __init__(self, t1:Track, t2: Track, prob: float, ecef: NDArray, lla: NDArray,
                 time: float, grid: NDArray, t1_grid: NDArray, t2_grid: NDArray, result_grid: NDArray) -> None:
        self.track1 = t1
        self.track2 = t2
        self.prob = prob
        self.pos_ecef = ecef
        self.pos_lla = lla
        self.time = time

        self.grid = grid
        self.t1_grid = t1_grid
        self.t2_grid = t2_grid
        self.result_grid = result_grid

        if t1 and t2:
            self.name = f"{t1.track_id}_{t2.track_id}_{time}"
        else:
            self.name = None


    def __repr__(self) -> str:
        return f"Conflict: {self.name}. Prob: {self.prob}. Time: {self.time}. Pos: {self.pos_lla}"


    def save(self) -> Dict:
        hist_dict : Dict[str, Union[int, str]] = dict()

        hist_dict["type"] = "conflict"

        hist_dict["name"] = f"{self.track1.track_id}_{self.track2.track_id}_{self.time}"
        hist_dict["track1"] = base64.b64encode(pickle.dumps(np.array(self.track1))).decode()
        hist_dict["track2"] = base64.b64encode(pickle.dumps(np.array(self.track2))).decode()
        hist_dict["pos_ecef"] = base64.b64encode(pickle.dumps(np.array(self.pos_ecef))).decode()
        hist_dict["pos_lla"] = base64.b64encode(pickle.dumps(np.array(self.pos_lla))).decode()
        hist_dict["time"] = base64.b64encode(pickle.dumps(np.array(self.time))).decode()
        hist_dict["prob"] = base64.b64encode(pickle.dumps(np.array(self.prob))).decode()
        hist_dict["grid"] = base64.b64encode(pickle.dumps(np.array(self.grid))).decode()
        hist_dict["t1_grid"] = base64.b64encode(pickle.dumps(np.array(self.t1_grid))).decode()
        hist_dict["t2_grid"] = base64.b64encode(pickle.dumps(np.array(self.t2_grid))).decode()
        hist_dict["result_grid"] = base64.b64encode(pickle.dumps(np.array(self.result_grid))).decode()

        return hist_dict


    def read(self, hist_dict):
        self.name = hist_dict["name"]
        #self.track1 = pickle.loads(base64.b64decode(hist_dict["track1"]))
        #self.track2 = pickle.loads(base64.b64decode(hist_dict["track1"]))
        self.pos_ecef = pickle.loads(base64.b64decode(hist_dict["pos_ecef"]))
        self.pos_lla = pickle.loads(base64.b64decode(hist_dict["pos_lla"]))
        self.time = pickle.loads(base64.b64decode(hist_dict["time"]))
        self.prob = pickle.loads(base64.b64decode(hist_dict["prob"]))
        self.grid = pickle.loads(base64.b64decode(hist_dict["grid"]))
        self.t1_grid = pickle.loads(base64.b64decode(hist_dict["t1_grid"]))
        self.t2_grid = pickle.loads(base64.b64decode(hist_dict["t2_grid"]))
        self.result_grid = pickle.loads(base64.b64decode(hist_dict["result_grid"]))


class ConflictDetector():
    def __init__(self, gate_distance: float, check_times: List[float], prob_threshold: float) -> None:
        self.gate_distance = gate_distance
        self.check_times = check_times
        self.prob_threshold = prob_threshold

        self.H: NDArray = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 1, 0, 0]])
        

    def _make_grid(self, midpoint: NDArray) -> NDArray:
        x_width = 5000
        y_width = 5000
        z_width = 5000

        x_grid = np.linspace(-(x_width / 2), x_width / 2, 4)
        y_grid = np.linspace(-(y_width / 2), y_width / 2, 4)
        z_grid = np.linspace(-(z_width / 2), z_width / 2, 4)

        base_grid = np.stack((x_grid, y_grid, z_grid))
        translate_grid = base_grid + midpoint.reshape((3, -1))

        xs, ys, zs = translate_grid

        search_grid = np.vstack(np.meshgrid(xs, ys, zs)).reshape(3,-1)

        return search_grid
    

    def detect(self, tracks: List[Track]) -> List[Conflict]:
        potential_conflicts: List[Conflict] = []

        for t1 in tracks:
            for t2 in tracks:
                if t1.track_id == t2.track_id:
                    # dont self collide
                    continue

                t1_pos = t1.filter.get_position()[0]
                t2_pos = t2.filter.get_position()[0]

                pos_delta = t1_pos - t2_pos
                midpoint = t2_pos + (pos_delta / 2)

                grid = self._make_grid(midpoint)
                
                if la.norm(pos_delta) <= self.gate_distance:
                    for t in self.check_times:
                        time = t1.filter.update_time + t
                        t1_pred_x, t1_pred_P = t1.predict(time)
                        t2_pred_x, t2_pred_P = t1.predict(time)

                        t1_pred_pos = self.H @ t1_pred_x
                        t1_pos_P = self.H @ t1_pred_P @ self.H.T

                        t2_pred_pos = self.H @ t2_pred_x
                        t2_pos_P = self.H @ t2_pred_P @ self.H.T                       

                        t1_dist = mvn(t1_pred_pos, t1_pos_P)
                        t2_dist = mvn(t2_pred_pos, t2_pos_P)

                        search_pts = grid.shape[1]

                        t1_grid = np.zeros(search_pts)
                        t2_grid = np.zeros(search_pts)

                        for i in range(search_pts):
                            t1_grid[i] = t1_dist.pdf(grid[:, i])
                            t2_grid[i] = t2_dist.pdf(grid[:, i])

                    t1_grid /= t1_grid.sum()
                    t2_grid /= t2_grid.sum()

                    result_grid = t1_grid * t2_grid
                    result_max = result_grid.max()

                    if result_max >= self.prob_threshold:
                        result_loc = grid[:, np.argmax(result_max)]
                        result_lla = pymap3d.ecef2geodetic(result_loc[0], result_loc[1], result_loc[2])

                        potential_conflicts.append(Conflict(t1, t2, result_max, result_loc, result_lla, time, grid, t1_grid, t2_grid, result_grid))
                        
        return potential_conflicts