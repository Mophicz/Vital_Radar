import numpy as np


# constants
from scipy.constants import c 
B = 1.7e9      


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
    

def slowVar(signal_matrix):
    """
    Calculates the slow-time variance of a signal matrix

    """
    signal_matrix = signal_matrix[-50:, :, :] 
    
    # calculate slow-time variance of the signal matrix
    var = None
    
    if len(signal_matrix) >= 2:
        # get variance along slow-time axis
        var = np.var(signal_matrix, axis=0, ddof=0)
        
        # sum the variances of the different antennas
        var = np.sum(var, axis=1)
        
    return var


def distance(var):
    # find the sample (frequency) bin with the highest variance
    idx = np.argmax(var)
        
    # convert sample (frequency) bin to range bin
    return sample2range(idx)
