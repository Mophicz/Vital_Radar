import numpy as np

def downsample(x, Fs, Fc, B):
    N = len(x)
    n = np.arange(N)
    x_bb = x * np.exp(-1j * 2 * np.pi * Fc * n / Fs)
    Xbb = np.fft.fft(x_bb)
    M = int(np.round(N * B / Fs))
    half_M = M // 2
    Xbb_shifted = np.fft.fftshift(Xbb)
    center = N // 2
    start = center - half_M
    end = center + half_M + 1
    Y = Xbb_shifted[start:end]
    y_bb_ds = np.fft.ifft(Y) * (M + 1) / N
    return y_bb_ds