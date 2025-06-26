from enum import Enum

import numpy as np

from vital_radar.processing.distance_estimation import slowVar, distance
from vital_radar.processing.beamformer import DelaySumBeamformer
from vital_radar.walabot.antenna_layout import antenna_layout


# constants
K = 137
F_START = 6.3e9
F_STOP = 8e9


class DisplayMode(Enum):
    RAW = 1
    IQ = 2
    DISTANCE = 3
    BREATHING = 4


def computePlotData(signal_matrix, display_mode, pairs=None):
    match display_mode:
        case DisplayMode.RAW | DisplayMode.IQ:
            # returns last signal
            data = signal_matrix[-1, :, 0]
            return  data / data.max()
            
        case DisplayMode.DISTANCE:
            # calculate slow time variance
            return slowVar(signal_matrix)
        
        case DisplayMode.BREATHING:
            # get antenna coordinates
            pos, pairs = antenna_layout.get_channel_positions(pairs)
            
            # construct array of frequency steps
            freqs = np.linspace(F_START, F_STOP, K) 
            
            # beamformer given these positions and frequencies
            bf = DelaySumBeamformer(pos, freqs)
            
            # estimate distance using variance method
            var = slowVar(signal_matrix)
            d = distance(var)
            
            # construct beamforming target from the distance
            r1 = np.array([0, 0, d])
            r2 = np.array([5, 5, d])
            r3 = np.array([5, -5, d])
            r4 = np.array([-5, 5, d])
            r5 = np.array([-5, -5, d])
            
            # apply beamformer
            B1 = bf.beamform(signal_matrix, r1)
            B2 = bf.beamform(signal_matrix, r2)
            B3 = bf.beamform(signal_matrix, r3)
            B4 = bf.beamform(signal_matrix, r4)
            B5 = bf.beamform(signal_matrix, r5)
            B = B1 + B2 + B3 + B4 + B5
            
            # collapse to slow-time
            x = np.abs(B).sum(axis=1)
    
            return x
        