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


if __name__ == "__main__":
    # start walabot
    init_walabot()
    
    # actual measurement
    signals, fs = measure()
    
    # flatten into 2‑D
    slow_time, range_profile, pairs = signals.shape
    flat = signals.reshape(slow_time, range_profile * pairs)

    # column names
    pairs    = [f"{a}-{b}" for a,b in [(1,2),(1,6),(1,10),(1,14)]]
    ranges = list(range(range_profile))
    cols     = pd.MultiIndex.from_product([pairs, ranges], names=["pair","range"])
    df       = pd.DataFrame(flat, index=np.arange(slow_time), columns=cols)
    df.index.name = "time"

    # write out CSV, with fs as the very first line
    out_path = "./data/radar_data_3m.csv"
    with open(out_path, "w") as f:
        # write a comment‐style header with fs
        f.write(f"# fs = {fs}\n")
        # now dump the DataFrame
        df.to_csv(f)
    
    # stop walabot
    stop_walabot()
       