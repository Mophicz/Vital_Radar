import numpy as np


# constants
FS = 102.4e9  # sampling frequency
FC = 7.15e9    # carrier frequency
B = 1.7e9      # bandwidth


def downconvert(x):
    """
    Downconversion of a signal at carrier frequency FC to baseband.
    Handles both 1D (single signal) and 2D input.
    """
    N = x.shape[0]
    n = np.arange(N)

    carrier = np.exp(-1j * 2 * np.pi * FC * n / FS).reshape(-1, 1)
    
    x_baseband = x * carrier
    return x_baseband


def downsample(x):
    """
    Performs downsampling of a signal by truncating around -bandwidth to +bandwidth in frequency domain
    Handles both 1D and 2D input
    """
    N = x.shape[0]
    
    # transform to frequency domain
    X = np.fft.fft(x, axis=0)
    
    # calculate the number of samples in the radar bandwidth
    M = int(np.round(N * B / FS))
    
    # shift zero to middle of frequency axis
    X_shifted = np.fft.fftshift(X, axes=0)
    
    # truncate, so that only the radar bandwidth is left
    center = N // 2
    half_M = M // 2
    start = center - half_M
    end = center + half_M + 1
    Y_truncated = X_shifted[start:end, :]
    
    # transform back to time domain
    y_downsampled = np.fft.ifft(Y_truncated, axis=0) * (M + 1) / N   
    return y_downsampled


def downsample_raw(x, factor):
    """
    Simple downsampling of the raw RF signal by integer factor, without downconversion.
    Keeps every nth sample to reduce resolution and computational load.
    Handles both 1D and 2D input.
    """
    x = np.array(x)
    if x.ndim == 1:
        return x[::factor]
    elif x.ndim == 2:
        return x[::factor, :]
    else:
        raise ValueError("Input must be 1D or 2D array")
    
    
def processRawSignal(x):
    """
    Processes raw signals (multiple in columns) from the Walabot API to numpy array,
    downconverted to baseband and downsampled.
    """
    x = np.array(x)
    x_bb = downconvert(x)
    x_ds = downsample(x_bb)
    return x_ds
    #return moving_average(x_ds, 20)
