import numpy as np
from scipy.ndimage import uniform_filter1d


# constants
FS = 102.4e9  # sampling frequency
FC = 7.15e9    # carrier frequency
B = 1.7e9      # bandwidth


def getStack(dq):
    """
    Converts a deque collection to a numpy array by stacking the elements.
    In this use case, elements are 2D-slices (fast-time x channels). Stacking them adds the slow-time dimension.
    
    Returns:
        signal_matrix: 3D numpy array (slow-time x fast-time x channels)
    """
    signal_matrix = np.stack(dq, axis=0)
    
    return signal_matrix


def dummy_signal_generator(freq=0.1, shape=(8192, 4)):
    """
    Infinite generator: each call to next(...) returns a new array: noise + sinusoidal phase.
    
    """
    while True:
        noise = np.random.normal(scale=0.1, size=shape)
        
        phase = np.zeros(shape)
        for channel in range(shape[1]):
            phase[:, channel] = np.sin(2 * np.pi * (channel+1) * freq * np.arange(shape[0]))
            
        signal = noise + phase
        yield signal


def moving_average(signal_matrix, window_size):
    """
    Applys a moving average to signal matrix along slow-time
  
    """
    # smooth along slow-time
    smoothed = uniform_filter1d(
        signal_matrix,
        size=window_size,
        axis=0,
        mode='nearest'
    )
    return smoothed
