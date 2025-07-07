import numpy as np
from scipy.signal import butter, filtfilt, welch, freqz
from statsmodels.regression.linear_model import yule_walker


def getWelch(x, fs, nfft=512):
    """
    Estimate the PSD of x using Welchs method.

    """
    default_nperseg = 128
    default_noverlap = 64
    
    # If the data is shorter than nperseg, shrink nperseg to len(x)
    nperseg = min(default_nperseg, len(x))
    
    # Make sure overlap is < nperseg
    noverlap = min(default_noverlap, nperseg - 1) if nperseg > 1 else 0
    
    f, P = welch(x, fs=fs,
             window='hann',
             nperseg=nperseg,     # ← from 16 up to 128 or more
             noverlap=noverlap,     # ← 50% overlap is common
             nfft=nfft,        # ← zero‑pad to get even finer grid
             average='median')
    
    return f, P


def getARpsd(x, fs, order=8, nfft=512):
    """
    Estimate the PSD of x using an AR fit of given order.

    """
    # 1) Fit AR model via Yule–Walker
    #    rho: AR coefficients (without the leading 1)
    #    sigma2: estimated white‐noise variance
    rho, sigma2 = yule_walker(x, order=order, method='mle')
    
    # 2) Build the full filter A(z) = 1 - sum_{k=1}^p rho[k-1] z^-k
    #    (statsmodels returns coefficients so that x[n] + sum rho*x[n-k] = noise)
    a = np.concatenate(([1.0], -rho))
    
    # 3) Compute frequency response of 1/A(z)
    #    We use freqz on the denominator 'a', numerator = [1].
    w, h = freqz(b=[1.0], a=a, worN=nfft, fs=fs)
    
    # 4) PSD = sigma2 * |H(e^{jω})|^2
    Pxx = sigma2 * (np.abs(h) ** 2)
    
    # Return only the one‐sided spectrum up to fs/2
    half = nfft // 2
    return w[:half], Pxx[:half]


def bandpassFilter(x, fs, lowcut=0.1, highcut=0.5, order=4):
    """
    Bandpass-filter x between lowcut and highcut (Hz) using an Nth-order Butterworth.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, x)
