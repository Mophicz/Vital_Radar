import numpy as np

from vital_radar.processing.utils import getStack


# constants
from scipy.constants import c # lightspeed
B = 1.7e9      # bandwidth


def sample2range(n):
    return (n*c) / (2*B)


def threshhold(x, threshhold):
    if x is not None:
        threshold_value = (threshhold / 100) * np.max(x)
        
        # Set values below threshold_value to zero
        result = np.where(x < threshold_value, 0, x)
        
        return result
    else:
        return x
    

def slowVar(s, threshhold):
    """
    Calculates the slow-time variance of a signal matrix

    """
    # calculate slow-time variance of the signal matrix
    var = None
    d = None
    
    if len(s) >= 2:
        # since s is a deque collection, the entries need to be stacked to an array first
        arr = getStack(s)
        
        # get variance along slow-time axis
        v = np.var(arr, axis=0, ddof=0)
        
        # normalize by dividing by expectation
        E = np.mean(arr)
        var = v / E
        
        # sum the variances of the different antennas
        var = np.sum(var, axis=1)

        # apply threshhold
        var = threshhold(var, threshhold)
        
        # find the sample (frequency) bin with the highest variance
        idx = np.argmax(var)
        
        # convert sample (frequency) bin to range bin
        d = sample2range(idx)
        
    return var, d
