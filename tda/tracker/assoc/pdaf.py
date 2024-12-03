import logging
import numpy as np
from scipy.stats import chi2
import sys
from typing import Callable, List, Sequence, Tuple

from .associator import Associator
from ..filters.filter import Filter
from ..initeator.initieator import Initeator
from ..initeator.truth_initieator import TruthIniteator
from ..track import Track
from .truth import TruthAssociator
from tda.common.measurement import Measurement


class PDAF(Associator, Initeator):
    def __init__(self, prob_gate: float, clutter_rate: float, init_updates: int, initial_initor: str, initial_associator: str, filter_factory: Callable[[Measurement], Filter ]):
        self.prob_gate = prob_gate
        self.gamma: float = -1.0  # chi sq cv for gating based on prob_gate
        self.clutter_rate = clutter_rate
        self.init_updates = init_updates  # number of steps to run the initial initiator and associator before beginning PDA
        self.frame_num = 0

        if initial_initor == "truth":
            self.initial_initor = TruthIniteator(filter_factory)
        else:
            logging.error(f"Unknown initator type \"{initial_initor}\". Exiting.")
            sys.exit(-1)

        if initial_associator == "truth":
            self.initial_associator = TruthAssociator()
        else:
            logging.error(f"Unknown initator type \"{initial_initor}\". Exiting.")
            sys.exit(-1)


    def initeate_tracks(self, frame: Sequence[Measurement]) -> List[Track]:
        self.frame_num += 1
        
        # we won't init any tracks after the initial period
        if self.frame_num > self.init_updates:
            return list()
        
        return self.initial_initor.initeate_tracks(frame)
    

    def assoceate(self, frame: Sequence[Measurement], tracks: Sequence[Track]) -> Tuple[List[Track], List[Measurement]]:
        # use initial assoceation routine until we've initalized
        if self.frame_num <= self.init_updates:
            return self.initial_associator.assoceate(frame, tracks)

        # do PDA routine if we have already initialized
        if len(tracks) != 1:
            logging.error(f"PDA can only handle a single track. Was passed {len(tracks)} tracks.")
            sys.exit(-1)

        track = tracks[0]
        pred_time = frame[0].time

        # if not yet computed, set gamma as a chi2 cv based on the provided prob gate
        if self.gamma < 0:
            self.gamma = chi2.ppf(self.prob_gate, frame[0].y.shape[0])

        # find meas within gate
        gated_meas = list()

        for m in frame:
            if track.meas_distance(m) < self.gamma:
                meas_likeli = track.meas_likelihood(m) * m.sensor_pd / self.clutter_rate
                gated_meas.append((m, meas_likeli))

        print(f"total gated: {len(gated_meas)}")

        # create combined inovation
        z_hat = track.predict_meas(pred_time)
        z_combined = np.zeros_like(z_hat)

        total_likeli = sum([likeli for _, likeli in gated_meas])
        beta_denom = (1 - (m.sensor_pd * self.prob_gate) + total_likeli)
        innov_i = list()
        for m, likeli in gated_meas:
            beta_i = likeli / beta_denom
            z_hat_i = m.y - z_hat  # innovation for this particular measurement
            z_combined += beta_i * z_hat_i

            innov_i.append((beta_i, z_hat_i))

        beta_0 = (1 - frame[0].sensor_pd * self.prob_gate) / beta_denom

        # compute weight and convariances
        x_pred, P_pred = track.predict(pred_time)
        K = track.compute_gain(pred_time)

        x_hat = x_pred + K @ z_combined
        P_c = P_pred - K @ track.compute_S(pred_time) @ K.T
                
        if len(gated_meas):
            innov_outer = np.outer(z_combined, z_combined)
            P_tilde = K @ np.sum([beta_i * np.outer(z_hat_i, z_hat_i) - innov_outer for beta_i, z_hat_i in innov_i], axis=0) @ K.T
        else:
            P_tilde = np.zeros_like(P_c)
            
        P = beta_0 * P_pred + (1 - beta_0) * P_c + P_tilde

        track.update_external(x_hat, P, pred_time)

        ungated_meas = [m for m in frame if m not in gated_meas]

        return list(), ungated_meas
        