import numpy as np


def getStack(dq):
    """
    Converts a deque collection to a numpy array by stacking the elements.
    In this use case, elements are 2D-slices (fast-time x channels). Stacking them adds the slow-time dimension.
    
    Returns:
        signal_matrix: 3D numpy array (fast-time x slow-time x channels)
    """
    signal_matrix = np.stack(dq, axis=0)
    
    return signal_matrix
