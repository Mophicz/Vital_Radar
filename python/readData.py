import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.constants import c

from antenna_layout import antenna_layout
from beamformer import DelaySumBeamformer
from distance_estimation import slowVar, distance
from spectrum_estimation import getWelch, bandpassFilter
from utils import moving_average


# constants
K = 137             # number frequency steps
F_START = 6.3e9     # start freqeuncy
F_STOP = 8e9        # stop frequency


def readData(path):
    # data path
    # csv_path = "./data/radar_data_v3.csv"
    with open(path, "r") as f:
        header = f.readline().strip()
        
    # sampling rate from header
    fs = float(header.lstrip("# ").split("=")[1])

    # read dataframe
    df = pd.read_csv(
        path,
        comment="#",          
        header=[0,1],         
        index_col=0           
    )

    # reshape back into (time, range-profile, pairs)
    M, flat_cols = df.shape
    n_channels = 137
    n_pairs    = flat_cols // n_channels

    arr3d = df.values.reshape(M, n_channels, n_pairs)

    # sanity check
    # print("array shape:", arr3d.shape)
    # print("fs:", fs)
    return fs, arr3d


def processDataNoBeamformer(signal_matrix, idx):
    
    x = signal_matrix.sum(axis=2)
    
    x = x[:, idx]
    
    return x
 
 
def processData1Point(signal_matrix, r):
    selected_pairs = [(1,2), (1,6), (1,10), (1,14)]
    
    pos, pairs = antenna_layout.get_channel_positions(selected_pairs)
    
    # construct array of frequency steps
    freqs = np.linspace(F_START, F_STOP, K) 
            
    # beamformer given these positions and frequencies
    bf = DelaySumBeamformer(pos, freqs)

    B = np.abs(bf.beamform(signal_matrix, r))
    
    x = B.sum(axis=1)
    
    return x
   

def processData(signal_matrix, points):
    selected_pairs = [(1,2), (1,6), (1,10), (1,14)]
    
    pos, pairs = antenna_layout.get_channel_positions(selected_pairs)
    
    # construct array of frequency steps
    freqs = np.linspace(F_START, F_STOP, K) 
            
    # beamformer given these positions and frequencies
    bf = DelaySumBeamformer(pos, freqs)

    B = np.zeros((points.shape[0], signal_matrix.shape[0], signal_matrix.shape[1]))
    # Beamform over all points:
    for i, r in enumerate(points):
        B[i, :] = np.array(np.abs(bf.beamform(signal_matrix, r)))
    
    # sum beams
    B_sum = B.sum(axis=0)  
    
    x = np.abs(B_sum).sum(axis=1)
    
    return x


def processData2(signal_matrix, points):
    selected_pairs = [(1,2), (1,6), (1,10), (1,14)]
    
    pos, pairs = antenna_layout.get_channel_positions(selected_pairs)
    
    # construct array of frequency steps
    freqs = np.linspace(F_START, F_STOP, K) 
            
    # beamformer given these positions and frequencies
    bf = DelaySumBeamformer(pos, freqs)

    B = np.zeros((points.shape[0], signal_matrix.shape[0], signal_matrix.shape[1]), dtype=np.complex64)
    # Beamform over all points:
    for i, r in enumerate(points):
        B[i, :] = np.array(bf.beamform(signal_matrix, r))
        
    # sum beams
    B_sum = B.sum(axis=0)  
    
    return B_sum.sum(axis=1)


def generate_grid(distance, radius, N):
    # 1D coordinates
    xs = np.linspace(-radius, radius, N)
    ys = np.linspace(-radius, radius, N)
    # 2D mesh
    xv, yv = np.meshgrid(xs, ys, indexing='xy')
    # flatten and stack
    x_flat = xv.ravel()
    y_flat = yv.ravel()
    z_flat = np.full_like(x_flat, distance)
    points = np.vstack((x_flat, y_flat, z_flat)).T
    return points


import numpy as np
from numpy import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

        
def visualize_grid(points, fov_elevation=None, fov_azimuth=None, max_distance=None):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    points = np.column_stack((points[:, 1], points[:, 2], points[:, 0]))
    
    elevation = points[:, 0]
    azimuth = points[:, 1]
    distance = points[:, 2]

    # Plot target points with legend
    ax.scatter(elevation[0:1], azimuth[0:1], distance[0:1], s=5, c='b', alpha=0.6, label='Zielpunkte')
    ax.scatter(elevation[1:], azimuth[1:], distance[1:], s=5, c='b', alpha=0.6)

    p1, p2 = points[9], points[14]
    #ax.quiver(p1[0], p1[1], p1[2], 
     #     p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2], 
      #    color='red', linewidth=1, arrow_length_ratio=0.2, normalize=False)

    midpoint = (p1 + p2) / 2
    dist = np.linalg.norm(p2 - p1)
    ax.text(midpoint[0]+0.01, midpoint[1], midpoint[2]+0.02, '5 cm', color='red', fontsize=9, ha='center')
    
    # Radar origin
    ax.scatter(0, 0, 0, color='r', s=50, label='Radar Position')
    ax.text(0, 0, -0.03, '(0,0,0)', color='red', fontsize=10, ha='center', va='center')

    # Connect each point to origin
    for e, a, d in zip(elevation, azimuth, distance):
        ax.plot([0, e], [0, a], [0, d], color='gray', linewidth=0.5, alpha=0.5)

    # Draw a double-sided red arrow below the point cloud
    arrow_length = np.linalg.norm(points[12])  # length of the arrow

    # Arrow goes from (0,0,z_offset) up and down
    #ax.quiver(-0.55, 0, 0, 0, 0, arrow_length, color='red', linewidth=1, arrow_length_ratio=0.05)
    #ax.quiver(-0.55, 0, 0 + arrow_length, 0, 0, -arrow_length, color='red', linewidth=1, arrow_length_ratio=0.05)

    # Label 'd' beside the arrow (shifted in X to avoid overlap)
    edge_point = points[20]
    m = edge_point[0]
    ax.text(m+0.02, arrow_length/2 , -m, f'd = {arrow_length:.2f} m', color='red', fontsize=10, ha='left', va='center')

    ax.legend()

    ax.set_zlabel('X: Höhe [m]')
    ax.set_xlabel('Y: Breite [m]')
    ax.set_ylabel('Z: Tiefe [m]')

    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    ax.zaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    
    a = Arrow3D([m, m], [0, arrow_length], [-m, -m],  mutation_scale=10, lw=2, arrowstyle="<|-|>", color="r")
    ax.add_artist(a)
    
    b = Arrow3D([p1[0], p2[0]], [p1[1], p2[1]], 
                [p1[2], p2[2]], mutation_scale=8, 
                lw=1, arrowstyle="<|-|>", color="r")
    ax.add_artist(b)

    # Axis limits
    if fov_elevation is not None:
        ax.set_zlim(fov_elevation[0], fov_elevation[1])
    if fov_azimuth is not None:
        ax.set_xlim(fov_azimuth[0], fov_azimuth[1])
    if max_distance is not None:
        ax.set_ylim(0, max_distance)

    #ax.set_xlim(ax.get_xlim()[::-1])  # Flip X-axis
    #ax.set_ylim(ax.get_ylim()[::-1])  # Flip Y-axis


    #ax.view_init(elev=-50, azim=-93, roll=-87.3)
    plt.savefig("Zielpunkte_plot.pdf", format="pdf", bbox_inches='tight', pad_inches=0.3)

    plt.show()
    

def plot(x, fs):
    fig, (ax_time, ax_psd) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    font = 12
    # remove DC content
    # x = x - np.mean(x)
    
    # apply moving average
    x = moving_average(x, 30)
    
    x = signal.detrend(x, type='linear')
    
    fc = 0.1
    b, a = signal.butter(2, fc/(fs/2), btype='high')
    x = signal.filtfilt(b, a, x)
    
    fc = 0.6
    b, a = signal.butter(2, fc/(fs/2), btype='low')
    x = signal.filtfilt(b, a, x)
    
    # FFT & PSD
    f, P = getWelch(x, fs)

    k = np.arange(0, len(x), 1)

    # Plot time-domain signal
    ax_time.plot(k/fs, x)
    ax_time.set_xlabel('Zeit [s]', fontsize=font)
    ax_time.set_ylabel('Amplitude', fontsize=font)

    # Plot frequency-domain PSD (normalized)
    ax_psd.plot(f, P)
    ax_psd.set_xlabel('Frequenz [Hz]', fontsize=font)
    ax_psd.set_ylabel('PSD', fontsize=font)
    ax_psd.set_xlim(0, 1)
    
    ax_psd.axvline(0.2, color='red', linestyle='--', label='Normbereich Atemfrequenz')
    ax_psd.axvline(0.3, color='red', linestyle='--')
    
    ax_psd.legend(fontsize=10)
    
    plt.savefig("breathing_signal.pdf", format='pdf', bbox_inches='tight')
    plt.show()


def plot2(x, fs, fc=7.15*1e9):
    """
    Plot the real part of the complex IQ signal and estimate displacement using phase differences.
    
    Parameters:
    -----------
    x : array_like
        Complex-valued, beamformed IQ signal over slow-time.
    fs : float
        Slow-time sampling rate (pulse repetition frequency, in Hz).
    fc : float
        Carrier frequency (in Hz).
        
    Returns:
    --------
    dx : ndarray
        Estimated displacement between successive pulses (in meters).
    """
    # Compute phase differences via conjugate product
    dphi = np.angle(x[1:] * np.conj(x[:-1]))
    # Unwrap to remove 2π jumps
    dphi = np.unwrap(dphi)
    # Convert phase shift to displacement
    lam = c / fc
    dx = (dphi / (4 * np.pi)) * lam
    
    # Time axis for displacement (one sample shorter)
    t_disp = np.arange(len(dx)) / fs
    
    # Plotting
    fig, (ax_time, ax_disp) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    
    ax_time.plot(np.real(x))
    ax_time.set_title('Real Part of IQ Signal')
    ax_time.set_xlabel('Sample Index')
    ax_time.set_ylabel('Amplitude')
    
    ax_disp.plot(t_disp, dx)
    ax_disp.set_title('Estimated Displacement Over Time')
    ax_disp.set_xlabel('Time [s]')
    ax_disp.set_ylabel('Displacement [m]')
    
    plt.show()
    
    return dx
 
 
def analyze_vitals(dx, fs, plot=False):
    """
    Analyze displacement signal for breathing rate and heart rate.
    
    Parameters
    ----------
    dx : array_like
        1-D displacement signal (meters).
    fs : float
        Sampling rate of dx in Hz.
    plot : bool, optional
        If True, plot PSD and filtered signals. Default is False.
        
    Returns
    -------
    results : dict
        Dictionary with keys:
        - 'breathing_rate_bpm': estimated breathing rate in breaths per minute
        - 'heart_rate_bpm': estimated heart rate in beats per minute
        - 'f_breathing': frequency array for breathing PSD
        - 'Pxx_breathing': PSD array for breathing band
        - 'f_heart': frequency array for heart PSD
        - 'Pxx_heart': PSD array for heart band
        - 'dx_breathing': bandpass-filtered breathing signal
        - 'dx_heart': bandpass-filtered heart signal
    """
    # Define frequency bands
    breathing_band = (0.1, 0.5)    # Hz (6 to 30 BPM)
    heart_band = (0.8, 2.0)        # Hz (48 to 120 BPM)
    
    # Bandpass filter design
    def bandpass(data, band):
        b, a = signal.butter(4, [band[0]/(fs/2), band[1]/(fs/2)], btype='band')
        return signal.filtfilt(b, a, data)
    
    dx_breathing = bandpass(dx, breathing_band)
    dx_heart = bandpass(dx, heart_band)
    
    # Compute PSD using Welch's method
    f_breath, Pxx_breath = signal.welch(dx_breathing, fs, nperseg=fs*4)
    f_heart, Pxx_heart = signal.welch(dx_heart, fs, nperseg=fs*4)
    
    # Find peak in each band
    idx_breath = np.argmax(Pxx_breath)
    idx_heart = np.argmax(Pxx_heart)
    
    f_breath_peak = f_breath[idx_breath]
    f_heart_peak = f_heart[idx_heart]
    
    # Convert to BPM
    breathing_rate_bpm = f_breath_peak * 60
    heart_rate_bpm = f_heart_peak * 60
    
    results = {
        'breathing_rate_bpm': breathing_rate_bpm,
        'heart_rate_bpm': heart_rate_bpm,
        'f_breathing': f_breath,
        'Pxx_breathing': Pxx_breath,
        'f_heart': f_heart,
        'Pxx_heart': Pxx_heart,
        'dx_breathing': dx_breathing,
        'dx_heart': dx_heart
    }
    
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        
        # Time-domain signals
        t = np.arange(len(dx)) / fs
        axes[0, 0].plot(t, dx_breathing)
        axes[0, 0].set_title('Breathing Signal (Bandpass 0.1-0.5 Hz)')
        axes[0, 0].set_xlabel('Time [s]')
        axes[0, 0].set_ylabel('Displacement [m]')
        
        axes[1, 0].plot(t, dx_heart)
        axes[1, 0].set_title('Heart Signal (Bandpass 0.8-2.0 Hz)')
        axes[1, 0].set_xlabel('Time [s]')
        axes[1, 0].set_ylabel('Displacement [m]')
        
        # PSDs
        axes[0, 1].semilogy(f_breath, Pxx_breath)
        axes[0, 1].set_title(f'Breathing PSD Peak = {breathing_rate_bpm:.1f} BPM')
        axes[0, 1].set_xlim(breathing_band)
        axes[0, 1].set_xlabel('Frequency [Hz]')
        axes[0, 1].set_ylabel('PSD')
        
        axes[1, 1].semilogy(f_heart, Pxx_heart)
        axes[1, 1].set_title(f'Heart PSD Peak = {heart_rate_bpm:.1f} BPM')
        axes[1, 1].set_xlim(heart_band)
        axes[1, 1].set_xlabel('Frequency [Hz]')
        axes[1, 1].set_ylabel('PSD')
        
        plt.show()
    
    return results
   
    

if __name__ == "__main__":
    fs, data = readData("../data/radar_data_3m.csv")
    
    # estimate distance using variance method
    var = slowVar(data)
    d = distance(var)
    idx = np.argmax(var)
    
    # Define grid
    points = generate_grid(d, 0.5, 5)
    r = np.array([0,0,d])
    
    #visualize_grid(points, fov_elevation=(-10, 10), fov_azimuth=(-10, 10), max_distance=10)
    
    #x = processDataNoBeamformer(data, idx)
    #x = processData1Point(data, r)
    x = processData(data, points)
    plot(x, fs)
  

"""
if __name__ == "__main__":
    fs, data = readData("../data/radar_data_3m.csv")
    
    # estimate distance using variance method
    var = slowVar(data)

    d = distance(var)
    

    
    # Define grid
    points = generate_grid(d, 0.1, 5)
    #points = np.vstack([(0,0,d),(5,5,d),(-5,5,d),(5,-5,d),(-5,-5,d)])
    r = np.array([0,0,d])
    
    #visualize_grid(points, fov_elevation=(-0.2, 0.2), fov_azimuth=(-0.2, 0.2), max_distance=3)
    
    #x = processDataNoBeamformer(data, idx)
    #x = processData1Point(data, r)
    x = processData2(data, points)
    
    dx = plot2(x, fs)
    
    #analyze_vitals(dx, fs, plot=True)
"""
"""
    n_peak = np.argmax(var)  # or given as an index
    peak_value = var[n_peak]
    plt.figure(figsize=(6, 4))
    plt.plot(var)
    plt.xlabel("fast-time-sample n", fontsize=14)
    plt.ylabel("Varianz", fontsize=14)
    plt.plot(n_peak, peak_value, 'ro')  # red dot
    plt.text(n_peak + 3, peak_value - 0.02 * max(var),
         r'$n_{\mathrm{peak}}$',
         color='red',
         fontsize=12,
         ha='left',
         va='center')
    plt.tight_layout(pad=0.1)  # ensure tight layout with minimal padding
    plt.savefig("variance_plot_python2.pdf", format='pdf', bbox_inches='tight')  # vector format, tight bounding box
    plt.show()
"""