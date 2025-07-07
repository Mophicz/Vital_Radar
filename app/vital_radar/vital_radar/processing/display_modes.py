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
            
            # set dead‑band threshold (in meters)
            d_threshold = 0.5   

            # initialize last updated distance
            last_update_d = None

            # beamformer given these positions and frequencies
            bf = DelaySumBeamformer(pos, freqs)
            
            # estimate distance using variance method
            var = slowVar(signal_matrix)
            d = distance(var)
            
            # Only when d moves by more than d_threshold from the last_update_d:
            if (last_update_d is None) or (abs(d - last_update_d) > d_threshold):
                # construct beamforming target from the distance
                r1 = np.array([0, 0, d])
                r2 = np.array([5, 5, d])
                r3 = np.array([5, -5, d])
                r4 = np.array([-5, 5, d])
                r5 = np.array([-5, -5, d])
                
                # compute & transpose steering weights per target
                W1 = bf._compute_weights(bf._compute_delays(r1)).T
                W2 = bf._compute_weights(bf._compute_delays(r2)).T
                W3 = bf._compute_weights(bf._compute_delays(r3)).T
                W4 = bf._compute_weights(bf._compute_delays(r4)).T
                W5 = bf._compute_weights(bf._compute_delays(r5)).T

                # cache the 5 weight matrices for reuse
                cached_weights = (W1, W2, W3, W4, W5)
                
                # reset last updated distance
                last_update_d = d
             
            W1, W2, W3, W4, W5 = cached_weights
                
            # multiply+sum for each beam
            B1 = (signal_matrix * W1[None, :, :]).sum(axis=2)
            B2 = (signal_matrix * W2[None, :, :]).sum(axis=2)
            B3 = (signal_matrix * W3[None, :, :]).sum(axis=2)
            B4 = (signal_matrix * W4[None, :, :]).sum(axis=2)
            B5 = (signal_matrix * W5[None, :, :]).sum(axis=2)
            
            # sum beams
            B = B1 + B2 + B3 + B4 + B5
            
            # collapse to slow time
            x = np.abs(B).sum(axis=1)
    
            return x
        