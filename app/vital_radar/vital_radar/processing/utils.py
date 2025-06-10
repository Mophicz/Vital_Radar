import numpy as np


def getStack(s):
    """
    Converts a deque collection to a numpy array by stacking the elements as columns next to eachother.
    
    """
    return np.stack(s, axis=0)
    