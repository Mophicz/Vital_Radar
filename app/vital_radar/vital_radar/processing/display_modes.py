from enum import Enum

import numpy as np

from vital_radar.processing.distance_estimation import slowVar, distance
from vital_radar.processing.beamformer import DelaySumBeamformer
from vital_radar.walabot.antenna_layout import antenna_layout


# constants
K = 137             # number frequency steps
F_START = 6.3e9     # start freqeuncy
F_STOP = 8e9        # stop frequency


class DisplayMode(Enum):
    """
    Adding a new element to this list adds a new element in the dropdown menu.
    
    """
    RAW = 1
    IQ = 2
    DISTANCE = 3
    BREATHING = 4


def computePlotData(signal_matrix, display_mode, pairs=None):
    """
    Defines the computation performed depending on the selected DisplayMode.
    
    """
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
            r2 = np.array([0.05, 0.05, d])
            r3 = np.array([0.05, -0.05, d])
            r4 = np.array([-0.05, 0.05, d])
            r5 = np.array([-0.05, -0.05, d])
                
            # multiply+sum for each beam
            B1 = bf.beamform(signal_matrix, r1)
            B2 = bf.beamform(signal_matrix, r2)
            B3 = bf.beamform(signal_matrix, r3)
            B4 = bf.beamform(signal_matrix, r4)
            B5 = bf.beamform(signal_matrix, r5)
            
            # sum beams
            B = B1 + B2 + B3 + B4 + B5
            
            # collapse to slow time
            x = np.abs(B).sum(axis=1)
    
            return x
        