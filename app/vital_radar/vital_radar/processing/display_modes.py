from enum import Enum

import numpy as np

from vital_radar.processing.distance_estimation import slowVar


class DisplayMode(Enum):
    RAW = 1
    DOWNSAMPLED = 2
    DISTANCE = 3


def computePlotData(signal_matrix, display_mode, threshhold=0):
        if display_mode == DisplayMode.RAW:
            return signal_matrix, None
        elif display_mode == DisplayMode.DISTANCE:
            var, d = slowVar(signal_matrix, threshhold)
            return var, d
        else:
            return np.zeros(1), None
            