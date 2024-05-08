import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from typing import Any, List, Optional, Tuple


class GroundTruth():
    def __init__(self) -> None:
        self.history: List[Tuple[float, NDArray]]=list()

    
    def update(self, state: NDArray, time: float) -> None:
        self.history.append((time, state))

    
    def get_state_hist(self, x_i: int) -> interp1d:
        n = len(self.history)
        xs = np.zeros(n)
        ts = np.zeros(n)

        for i, (time, state) in enumerate(self.history):
            xs[i] = state[x_i]
            ts[i] = time

        return interp1d(ts, xs)
    

    def plot_state_hist(self, x_is: Optional[List[int]]=None):
        if len(self.history) == 0:
            return None

        if x_is is None:
            x_is = list(range(len(self.history[0][1])))

        fig, axs = plt.subplots(len(x_is))

        ts = [t for (t, _) in self.history]

        for x_i in x_is:
            xs = [x[x_i] for (_, x) in self.history]

            axs[x_i].scatter(ts, xs)


        return fig, axs