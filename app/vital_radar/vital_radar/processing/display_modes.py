from enum import Enum

from vital_radar.processing.distance_estimation import slowVar


class DisplayMode(Enum):
    RAW = 1
    IQ = 2
    DISTANCE = 3


def computePlotData(signal_matrix, display_mode):
    match display_mode:
        case DisplayMode.RAW | DisplayMode.IQ:
            # returns last signal
            data = signal_matrix[-1, :, 0]
            return  data / data.max()
            
        case DisplayMode.DISTANCE:
            # calculate slow time variance
            return slowVar(signal_matrix)
        