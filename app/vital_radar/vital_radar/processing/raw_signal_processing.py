import numpy as np


# constants
FS = 102.4e9 # smapling frequency
FC = 7.15e9 # carrier frequency
B = 1.7e9 # bandwidth


def downconvert(x):
    """
    Downconversion of a signal at carrier frequency FC to baseband.
    
    """
    N = len(x)
    n = np.arange(N)
    
    # formula for downconversion to baseband
    x_baseband = x * np.exp(-1j * 2 * np.pi * FC * n / FS)
    return x_baseband


def downsample(x):
    """
    Performs downsampling of a signal by truncating around -bandwidth to +bandwidth in frequency domain

    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    N = len(x)
    
    # transform to frequency domain
    X = np.fft.fft(x)
    
    # calculate the number of samples in the radar bandwidth
    M = int(np.round(N * B / FS))
    
    # shift zero to middle of frequency axis
    X_shifted = np.fft.fftshift(X)
    
    # truncate, so that only the radar bandwidth is left
    center = N // 2
    half_M = M // 2
    start = center - half_M
    end = center + half_M + 1
    Y_truncated = X_shifted[start:end]
    
    # transform back to time domain
    y_downsampled = np.fft.ifft(Y_truncated) * (M + 1) / N   
    return y_downsampled


def process_raw_signal(x):
    """
    Processes a raw signal from the walabot API to be a numpy array, 
    that has been downconverted to baseband and downsampled to a lower 
    time-resolutoin corresponding to the number of distinct frequency steps
    used by the radar.

    """
    # to numpy array
    x = np.array(x)
    
    # downconvert and downsample
    x_bb = downconvert(x)
    x_ds = downsample(x_bb)
    
    return x_ds
