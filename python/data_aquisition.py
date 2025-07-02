import os
import time

import WalabotAPI as wlbt
import numpy as np
import pandas as pd

import signal_aquisition as sa
from raw_signal_processing import processRawSignal


def init_walabot():
    # Load the WalabotAPI library
    wlbt.Init()
    wlbt.Initialize()

    # Connect to the Walabot device
    wlbt.ConnectAny()

    # Set scanning profile to raw signal
    wlbt.SetProfile(wlbt.PROF_SENSOR)

    # Set dynamic image filter to none
    wlbt.SetDynamicImageFilter(wlbt.FILTER_TYPE_NONE)

    # Start the Walabot
    wlbt.Start()

    # Calibrate
    wlbt.StartCalibration()
    stat, prog = wlbt.GetStatus()
    while stat == wlbt.STATUS_CALIBRATING and prog < 100:
        wlbt.Trigger()
        stat, prog = wlbt.GetStatus()
        print(f"Calibrating {prog}%")
    print("Calibration complete")


def stop_walabot():
    # Stop and disconnect
    wlbt.Stop()
    wlbt.Disconnect()
    wlbt.Clean()


def get_raw_signals(antenna_pairs, N_st, N_ft):
    # initialize array for saving raw signals 
    signals = np.zeros((len(antenna_pairs), N_st, N_ft)) # shape: (40, 200, 8192) for all 40 antenna pairs and 200 slow time samples

    # start time for capturing measurement duration
    start = time.time()
    index = 1
    for i in range(0, N_st):
        
        wlbt.Trigger()
        
        for j, pair in enumerate(antenna_pairs):
            tx = pair[0]
            rx = pair[1]
            
            amplitudes, time_axis = wlbt.GetSignal(pair)

            signals[j, i, :] = amplitudes

            print(f"Index {index} | Trigger: {i} | Pair {j} | TX: {tx}, RX: {rx} | Samples: {len(amplitudes)}")
            index +=1
            
    # end time for capturing measurment duration
    end = time.time()
    
    # measurement duration
    t = end - start
    
    # slow time sampling frequency
    F_st = N_st / t
    
    # return signals array and slow time sampling frequency
    return signals, F_st


def save_signals(output_dir, filename, signals, F_st):
    # make sure directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # create file path
    file_path = os.path.join(output_dir, f"{filename}.npz")

    np.savez(file_path, signals=signals, F_st=F_st)

    print(f"Saved to {file_path}")


def measure(t=30):
    selected_pairs = [(1,2), (1,6), (1,10), (1,14)]
    
    n = np.arange(20)
    for i in n:
        signals = sa.getSignals(selected_pairs)
        
    fs = sa.trigger_freq
    M = int(np.ceil(t * fs))
    signal_buffer = np.zeros((M, 137, len(selected_pairs)))
    
    m = np.arange(M)
    for i in m:
        signals = sa.getSignals(selected_pairs)
        signals = processRawSignal(signals)
        signal_buffer[i, :, :] = signals
        print(i)
    print(sa.trigger_freq)
        
    return signal_buffer, sa.trigger_freq
    

N_SLOW_TIME = 200
N_FAST_TIME = 8192
OUTPUT_DIR = "C:\\Users\\Michael\\Projects\\Projektseminar_Medizintechnik\\Vital_Radar\\data"
FILENAME = "radar_data_v2"


if __name__ == "__main__":
    # start walabot
    init_walabot()
    
    # actual measurement
    signals, fs = measure()
    
    # 2) flatten into 2‑D so we can make a DataFrame
    M, n_ch, n_pairs = signals.shape
    flat = signals.reshape(M, n_ch * n_pairs)

    # 3) build nice column names
    pairs    = [f"{a}-{b}" for a,b in [(1,2),(1,6),(1,10),(1,14)]]
    channels = list(range(n_ch))
    cols     = pd.MultiIndex.from_product([pairs, channels], names=["pair","channel"])
    df       = pd.DataFrame(flat, index=np.arange(M), columns=cols)
    df.index.name = "sample"

    # 4) write out CSV, with fs as the very first line
    out_path = "signal_buffer_with_fs.csv"
    with open(out_path, "w") as f:
        # write a comment‐style header with fs
        f.write(f"# fs = {fs}\n")
        # now dump the DataFrame
        df.to_csv(f)
    
    # stop walabot
    stop_walabot()
    
    # save signals to numpy file
    
    