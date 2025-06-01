import os
import time

import WalabotAPI as wlbt
import numpy as np


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


def stop_walabot():
    # Stop and disconnect
    wlbt.Stop()
    wlbt.Disconnect()
    wlbt.Clean()


def get_raw_signals(antenna_pairs, N_st, N_ft):
    # initialize array for saving raw signals 
    signals = np.zeros(len(antenna_pairs), N_st, N_ft) # shape: (40, 200, 8192) for all 40 antenna pairs and 200 slow time samples

    # start time for capturing measurement duration
    start = time.time()

    for i in range(0, N_st):
        
        wlbt.Trigger()
        
        for j, pair in enumerate(antenna_pairs):
            tx = pair[0]
            rx = pair[1]
            
            amplitudes, time_axis = wlbt.GetSignal(pair)

            signals[j, i, :] = amplitudes

            print(f"Index {i+j} | Trigger: {i} | Pair {j} | TX: {tx}, RX: {rx} | Samples: {len(amplitudes)}")
    
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
    file_path = os.path.join(output_dir, filename, ".npz")

    np.savez(file_path, signals=signals, F_st=F_st)

    print(f"Saved to {file_path}")


N_SLOW_TIME = 200
N_FAST_TIME = 8192
OUTPUT_DIR = "C:\Users\Michael\Projects\Projektseminar_Medizintechnik\Vital_Radar\data"
FILENAME = "radar_data_v2"


if __name__ == "__main__":
    # start walabot
    init_walabot()
    
    # choose antenna pairs (here all)
    antenna_pairs = wlbt.GetAntennaPairs()
    
    # actual measurement
    signals, F_st = get_raw_signals(antenna_pairs, N_SLOW_TIME, N_FAST_TIME)
    
    # stop walabot
    stop_walabot()
    
    # save signals to numpy file
    save_signals(OUTPUT_DIR, FILENAME, signals, F_st)
    