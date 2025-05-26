import os
import h5py
import WalabotAPI as wlbt
import time
import numpy as np

# Load the WalabotAPI library
wlbt.Init()
wlbt.Initialize()

# Connect to the Walabot device
wlbt.ConnectAny()

# Set scanning profile
wlbt.SetProfile(wlbt.PROF_SENSOR)  # Raw signal mode

# Set dynamic image filter to none
wlbt.SetDynamicImageFilter(wlbt.FILTER_TYPE_NONE)

# Set arena parameters (dummy values; adjust as needed)
# These define the scanning volume in mm
wlbt.SetArenaR(30, 150, 1)
wlbt.SetArenaTheta(-20, 20, 10)
wlbt.SetArenaPhi(-45, 45, 5)

# Set to sensor mode
wlbt.SetThreshold(0.5)

# Start the Walabot
wlbt.Start()

# Calibrate (recommended before capturing signals)
wlbt.StartCalibration()

duration_seconds = 10
start_time = time.time()

output_dir = "walabot_signals"
os.makedirs(output_dir, exist_ok=True)  # Creates the directory if it doesn't exist

# Loop until the time has elapsed
frame_count = 0
signals = []

while time.time() - start_time < duration_seconds:
    wlbt.Trigger()
    antennaPairs = wlbt.GetAntennaPairs()
    frame_signals = []
    for pair in antennaPairs:
        tx = pair[0]
        rx = pair[1]
        timeAxis, amplitudes = wlbt.GetSignal(pair)

        frame_signals.append((timeAxis, amplitudes))

        print(f"Frame {frame_count+1} | TX: {tx}, RX: {rx} | Samples: {len(amplitudes)}")
    
    signals.append(frame_signals)
    frame_count += 1

signals = np.array(signals, dtype=object)
print(signals.shape)

output_dir = "walabot_signals"
os.makedirs(output_dir, exist_ok=True)
output_h5 = os.path.join(output_dir, "signals.h5")

with h5py.File(output_h5, 'w') as hf:
    for frame_idx, frame in enumerate(signals):
        grp = hf.create_group(f"frame_{frame_idx}")
        for pair_idx, (timeAxis, amplitudes) in enumerate(frame):
            # Ensure native NumPy float arrays
            timeAxis = np.array(timeAxis, dtype=np.float32)
            amplitudes = np.array(amplitudes, dtype=np.float32)

            pair_grp = grp.create_group(f"pair_{pair_idx}")
            pair_grp.create_dataset("timeAxis", data=timeAxis)
            pair_grp.create_dataset("amplitudes", data=amplitudes)

print(f"Saved HDF5 to {output_h5}")

# Stop and disconnect
wlbt.Stop()
wlbt.Disconnect()
wlbt.Clean()
