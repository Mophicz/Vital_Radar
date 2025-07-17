import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


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


def visualize_grid(points, fov_elevation=None, fov_azimuth=None, max_distance=None):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Swap axes for desired orientation: x=elevation, y=azimuth, z=distance
    elevation = points[:, 0]
    azimuth = points[:, 1]
    distance = points[:, 2]

    ax.scatter(elevation, azimuth, distance, s=5, c='b', alpha=0.6)

    # Add origin marker for radar position
    ax.scatter(0, 0, 0, color='r', s=50, label='Radar Origin')
    ax.legend()

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Beamforming Targets')

    # Set axis limits based on radar field of view (FOV) and range
    if fov_elevation is not None:
        ax.set_xlim(fov_elevation[0], fov_elevation[1])
    if fov_azimuth is not None:
        ax.set_ylim(fov_azimuth[0], fov_azimuth[1])
    if max_distance is not None:
        ax.set_zlim(0, max_distance)

    # Set custom view
    ax.view_init(elev=-50, azim=-93, roll=-87)

    plt.tight_layout()
    plt.show()
    

def plot(x, fs):
    fig, (ax_time, ax_psd) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    
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
    
    #x = bandpassFilter(x, fs)
    
    # FFT & PSD
    #P = np.fft.fft(x)
    #f = np.fft.fftfreq(len(x), d=1/fs)
    f, P = getWelch(x, fs)
    #f, P = getARpsd(x, fs)

    k = np.arange(0, len(x), 1)

    # Plot time-domain signal
    ax_time.plot(k/fs, x)
    
    ax_time.set_title('Time Signal')
    
    ax_time.set_xlabel('Time (s)')
    
    ax_time.set_ylabel('Amplitude')
    
    #ax_time.set_ylim(-0.005, 0.005)

    # Plot frequency-domain PSD (normalized)
    ax_psd.plot(f, P)
    
    ax_psd.set_title('Spectrum Estimate')
    
    ax_psd.set_xlabel('Frequency (Hz)')
    ax_psd.set_xlim(0, 1 )
    
    ax_psd.set_ylabel('Logarithmic PSD')

    ax_psd.axvline(0.2, color='red', linestyle='--', label='Expected Breathing Range')
    ax_psd.axvline(0.3, color='red', linestyle='--')
    
    ax_psd.legend()
    
    plt.show()

"""
if __name__ == "__main__":
    fs, data = readData("./data/radar_data_3m.csv")
    
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
    fs, data = readData("./data/radar_data_3m.csv")
    
    # estimate distance using variance method
    var = slowVar(data)
    d = distance(var)
    idx = np.argmax(var)
    
    # Define grid
    points = generate_grid(d, 0.5, 5)
    #points = np.vstack([(0,0,d),(5,5,d),(-5,5,d),(5,-5,d),(-5,-5,d)])
    r = np.array([0,0,d])
    
    visualize_grid(points, fov_elevation=(-10, 10), fov_azimuth=(-10, 10), max_distance=10)
    
    #x = processDataNoBeamformer(data, idx)
    #x = processData1Point(data, r)
    x = processData(data, points)
    plot(x, fs)
    