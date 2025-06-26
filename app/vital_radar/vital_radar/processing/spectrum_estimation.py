import numpy as np
import scipy.signal as sig


def getWelch(x, fs):
    return sig.welch(x, fs=fs, window='hann', nperseg=32, noverlap=8, average='median')
