import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------
# 1) Downsample‐to‐Radar‐Bandwidth Function
# ----------------------------------------------
def downsample(x_fast, Fs, Fc, B):
    """
    Converts a fast‐time signal x_fast (1D array, length N) to baseband,
    truncates to bandwidth B, then iDFT’s back. Matches the MATLAB logic:
      y = downsample(x, Fs, Fc, B)
    where:
      - x_fast:  length‐N complex or real waveform (fast time)
      - Fs:      fast‐time sampling frequency (e.g. 102.4e9)
      - Fc:      carrier frequency (e.g. 7.15e9)
      - B:       radar bandwidth (e.g. 1.7e9)
    Returns
      y_bb_ds:  length (M+1) complex waveform, where M = int(N * B/Fs)
    """
    N = x_fast.shape[0]
    n = np.arange(N)

    # 1) Downconvert to baseband
    x_bb = x_fast * np.exp(-1j * 2 * np.pi * Fc * n / Fs)

    # 2) DFT of baseband signal
    Xbb = np.fft.fft(x_bb)

    # 3) Determine number of samples in radar‐bandwidth
    M = int(np.round(N * B / Fs))  # → matches MATLAB: N * B/Fs exactly
    half_M = M // 2

    # 4) Truncate in frequency by centering and selecting M+1 bins
    Xbb_shifted = np.fft.fftshift(Xbb)
    center = N // 2
    start = center - half_M
    end = center + half_M + 1   # end is exclusive in Python slicing → yields M+1 points

    Y = Xbb_shifted[start:end]  # length = M+1

    # 6) iDFT and normalize
    y_bb_ds = np.fft.ifft(Y) * (M + 1) / N

    return y_bb_ds


if __name__ == "__main__":
  # ----------------------------------------------
  # 2) Load Raw Signals from .npz
  # ----------------------------------------------
  # Adjust this path to wherever you saved signals / F_st
  NPZ_PATH = r"C:\Users\Michael\Projects\Projektseminar_Medizintechnik\Vital_Radar\data\radar_data_v2.npz"

  data = np.load(NPZ_PATH)
  # 'signals' has shape (n_pairs, N_slow_time, N_fast_time)
  # 'F_st' is the slow‐time sampling frequency
  signals = data["signals"]  # shape example: (40, 200, 8192)
  F_st = data["F_st"].item()  # maybe a scalar

  # For all subsequent examples, we’ll just pick the first antenna‐pair index = 0
  pair_idx = 0
  # Extract 2D array for that pair: slow‐time along axis=1, fast‐time along axis=2
  # So x_raw[f] is the f‐th slow‐time fast‐time waveform
  sig_pair = signals[pair_idx, :, :]  # shape: (N_slow_time, N_fast_time)

  # ----------------------------------------------
  # 3) Example: Downsampling One Fast‐Time Waveform
  # ----------------------------------------------
  Fs = 102.4e9   # fast‐time sampling freq
  Fc = 7.15e9    # carrier freq
  B  = 1.7e9     # radar bandwidth

  # Pick the first slow‐time index f = 0:
  x = sig_pair[0, :]           # shape (N_fast_time,)  e.g. length 8192
  N = x.shape[0]
  n = np.arange(N)

  # (a) Downconvert to baseband
  x_bb = x * np.exp(-1j * 2 * np.pi * Fc * n / Fs)

  # (b) PSD of raw signal
  Pxx = np.abs(np.fft.fft(x)) ** 2 / (N * Fs)
  Pxx = np.fft.fftshift(Pxx)

  # (c) PSD of baseband signal
  Xbb = np.fft.fft(x_bb)
  Pbb = np.abs(Xbb) ** 2 / (N * Fs)
  Pbb = np.fft.fftshift(Pbb)

  # (d) Downsample baseband‐spectrum
  M = int(np.round(N * B / Fs))    # from above, M=136 for your parameters
  half_M = M // 2
  center = N // 2

  Xbb_shifted = np.fft.fftshift(Xbb)
  start = center - half_M
  end = center + half_M + 1        # length = M+1
  Xbb_ds = Xbb_shifted[start:end]  # truncated frequency bins
  Pbb_ds = np.abs(Xbb_ds) ** 2 / (N * Fs)

  # (e) iDFT to get downsampled baseband in time
  x_bb_ds = np.fft.ifft(Xbb_ds) * (M + 1) / N  # length M+1

  # (f) Build frequency axis (in GHz) for plots
  f = np.linspace(-Fs/2, Fs/2, N) / 1e9  # length N, in GHz

  # (g) Plot 2×3 panels exactly like MATLAB’s figure
  fig, axes = plt.subplots(2, 3, figsize=(12, 8))
  axes = axes.ravel()

  # Panel (1): Raw waveform
  axes[0].plot(x.real, label="Real")  # x is real anyway, but for consistency
  axes[0].set_xlim(0, 8200)
  axes[0].set_title("Raw (pair 0, slow‐time 0)")
  axes[0].set_xlabel("Fast‐Time Sample n")
  axes[0].set_ylabel("s_l(n)")

  # Panel (2): Baseband real & imag
  axes[1].plot(x_bb.real, label="Real")
  axes[1].plot(x_bb.imag, label="Imag", alpha=0.7)
  axes[1].set_xlim(0, 8200)
  axes[1].set_title("Baseband (pair 0, slow‐time 0)")
  axes[1].set_xlabel("Fast‐Time Sample n")
  axes[1].set_ylabel("s_l(n)")
  axes[1].legend()

  # Panel (3): Downsampled real & imag
  axes[2].plot(x_bb_ds.real, label="Real")
  axes[2].plot(x_bb_ds.imag, label="Imag", alpha=0.7)
  axes[2].set_xlim(0, M + 1 + 1)  # ~137 points
  axes[2].set_title("Downsampled (pair 0, slow‐time 0)")
  axes[2].set_xlabel("Fast‐Time Sample n")
  axes[2].set_ylabel("s_l(n)")
  axes[2].legend()

  # Panel (4): PSD of raw (shifted)
  axes[3].plot(f, 20 * np.log10(Pxx + 1e-20))  # add tiny value to avoid log(0)
  axes[3].set_xlim(0, 10)
  axes[3].set_xlabel("Frequency [GHz]")
  axes[3].set_ylabel(r"20 log $P_{ss}(e^{j\omega})$")

  # Panel (5): PSD of baseband (shifted)
  axes[4].plot(f, 20 * np.log10(Pbb + 1e-20))
  axes[4].set_xlim(-5, 5)
  axes[4].set_xlabel("Frequency [GHz]")
  axes[4].set_ylabel(r"20 log $P_{ss}(e^{j\omega})$")

  # Panel (6): PSD of downsampled baseband (shifted)
  # Build a frequency vector for the truncated bin range:
  f_ds = np.linspace(-Fs/2, Fs/2, N)[start:end] / 1e9
  axes[5].plot(f_ds, 20 * np.log10(Pbb_ds + 1e-20))
  axes[5].set_xlim(-5, 5)
  axes[5].set_xlabel("Frequency [GHz]")
  axes[5].set_ylabel(r"20 log $P_{ss}(e^{j\omega})$")

  plt.tight_layout()
  plt.savefig("FT_downsampling_python.png", dpi=300)
  #plt.show()


  # ----------------------------------------------
  # 4) SVD Decluttering Over All Slow‐Time Samples
  # ----------------------------------------------
  # We already have sig_pair of shape (N_slow_time, N_fast_time).
  N_slow_time, N_ft = sig_pair.shape
  M = int(np.round(N_ft * B / Fs))    # M=136
  y = np.zeros((M + 1, N_slow_time), dtype=complex)

  # Build the full downsampled fast‐time × slow‐time matrix y
  for idx_st in range(N_slow_time):
      x_ft = sig_pair[idx_st, :]   # length N_ft
      y[:, idx_st] = downsample(x_ft, Fs, Fc, B)

  # Perform SVD
  #   y has shape (M+1, N_slow_time). We want U*(S)*Vh such that y = U @ np.diag(S) @ Vh
  U, S, Vh = np.linalg.svd(y, full_matrices=False)

  # Choose k1=1 (DC) and k2=3 (VS)
  k1 = 1
  k2 = 3

  # DC component: rank‐1 (k1)
  U_DC = U[:, :k1]         # shape ((M+1)×1)
  S_DC = np.diag(S[:k1])   # shape (1×1)
  Vh_DC = Vh[:k1, :]       # shape (1×N_slow_time)
  X_DC = U_DC @ S_DC @ Vh_DC  # shape ((M+1)×N_slow_time)

  # VS component: ranks 2..3 (k1+1 to k2)
  U_VS = U[:, k1:k2]                       # shape ((M+1)×(k2−k1))
  S_VS = np.diag(S[k1:k2])                 # shape ((k2−k1)×(k2−k1))
  Vh_VS = Vh[k1:k2, :]                     # shape ((k2−k1)×N_slow_time)
  X_VS = U_VS @ S_VS @ Vh_VS               # shape ((M+1)×N_slow_time)

  # Noise component: ranks k2+1..end
  U_N = U[:, k2:]
  S_N = np.diag(S[k2:])
  Vh_N = Vh[k2:, :]
  X_N = U_N @ S_N @ Vh_N

  # Plot the three “imagesc” panels side by side
  fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))
  im0 = axes2[0].imshow(np.abs(X_DC), aspect="auto")
  axes2[0].set_title("X_DC (k=1)")
  plt.colorbar(im0, ax=axes2[0])

  im1 = axes2[1].imshow(np.abs(X_VS), aspect="auto")
  axes2[1].set_title("X_VS (k=2,3)")
  plt.colorbar(im1, ax=axes2[1])

  im2 = axes2[2].imshow(np.abs(X_N), aspect="auto")
  axes2[2].set_title("X_N (k>3)")
  plt.colorbar(im2, ax=axes2[2])

  plt.tight_layout()
  plt.savefig("SVD_declutter_python.png", dpi=300)
  # plt.show()


  # ----------------------------------------------
  # 5) Distance & Variance Plot
  # ----------------------------------------------
  c = 3e8      # m/s
  K = 137
  n_fac = 4
  N_ds = M + 1

  dF = B / K
  d = (n_fac * c) / (2 * N_ds * dF)
  print(f"Computed distance d = {d:.3e} m")

  v = np.var(y[:, :199], axis=1, ddof=0)

  # Plot v vs. index
  plt.figure(figsize=(6, 4))
  plt.plot(v)
  plt.xlabel("fast-time-sample n", fontsize=14)
  plt.ylabel("Varianz", fontsize=14)
  plt.tight_layout(pad=0.1)  # ensure tight layout with minimal padding
  plt.savefig("variance_plot_python2.pdf", format='pdf', bbox_inches='tight')  # vector format, tight bounding box
  plt.show()
