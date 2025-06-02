import numpy as np
import matplotlib.pyplot as plt

from raw_data_preprocessing import downsample


def mov(y, window_len):
    """
    Compute a centered moving average of length `window_len` along the slow‐time axis
    for each fast‐time row of y.

    Parameters
    ----------
    y : np.ndarray, shape = (M_plus_1, N_slow_time)
        Your downsampled‐baseband matrix.
    window_len : int
        The length of the moving‐average window (must be >= 1 and <= N_slow_time).

    Returns
    -------
    y_ma : np.ndarray, shape = (M_plus_1, N_slow_time)
        The moving‐averaged version of y along axis=1 (slow‐time), using mode='same' so
        that the output length matches the input.
    """
    if window_len < 1 or window_len > y.shape[1]:
        raise ValueError("window_len must be between 1 and N_slow_time")

    # Create a 1D kernel of length `window_len`, normalized to sum to 1
    kernel = np.ones(window_len) / window_len

    # Allocate output
    y_ma = np.zeros_like(y, dtype=float)

    # Convolve each fast‐time row independently
    for i in range(y.shape[0]):
        # mode='same' keeps the output length = input length,
        # centering the window (with partial windows at the edges).
        y_ma[i, :] = np.convolve(y[i, :], kernel, mode="same")

    return y_ma


if __name__ == "__main__":
    # load data
    NPZ_PATH = r"C:\Users\Michael\Projects\Projektseminar_Medizintechnik\Vital_Radar\data\radar_data_v2.npz"

    data = np.load(NPZ_PATH)

    signals = data["signals"]
    F_st = data["F_st"].item()  

    pair_idx = 0
    sig_pair = signals[pair_idx, :, :]

    Fs = 102.4e9   # fast‐time sampling freq
    Fc = 7.15e9    # carrier freq
    B  = 1.7e9     # radar bandwidth

    N_slow_time, N_ft = sig_pair.shape
    M = int(np.round(N_ft * B / Fs))

    y = np.zeros((M + 1, N_slow_time), dtype=complex)

    # Build the full downsampled fast‐time × slow‐time matrix y
    for idx_st in range(N_slow_time):
        x_ft = sig_pair[idx_st, :]   # length N_ft
        y[:, idx_st] = downsample(x_ft, Fs, Fc, B)
        
    print(y.shape)
    print(y.dtype)

    v = np.var(y[:, :30], axis=1, ddof=0)

    plt.figure(figsize=(6, 4))
    plt.plot(v)
    plt.title("Slow-time variance")
    plt.xlabel("Downsampled fast‐time index")
    plt.ylabel("Variance")
    plt.tight_layout()
    plt.show()