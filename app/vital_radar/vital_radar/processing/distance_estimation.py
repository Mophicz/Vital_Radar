import numpy as np

from vital_radar.processing.utils import getStack


def slowVar(s):
    """
    Calculates the slow-time variance of a signal matrix

    """
    # calculate slow-time variance of the signal matrix
    var = None
    if len(s) >= 2:
        # since s is a deque collection, the entries need to be stacked to an array first
        arr = getStack(s)
        # get variance along slow-time axis
        v = np.var(arr, axis=0, ddof=0)
        # normalize by dividing by expectation
        E = np.mean(arr)
        var = v / E
    
    return var


# TODO: Convert slow time variance frequency bins to distance bins and select peak as estimated distance
