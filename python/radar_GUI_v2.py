import sys
from collections import deque

import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QPushButton, QLabel, QHBoxLayout
)
from PyQt6.QtCore import QTimer, Qt, QThread
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import WalabotAPI as wlbt


# ----------------------------------------------
# 1) Downsample‐to‐Radar‐Bandwidth Function
# ----------------------------------------------
def downsample(x, Fs, Fc, B):
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
    N = len(x)
    n = np.arange(N)

    # 1) Downconvert to baseband
    x_bb = x * np.exp(-1j * 2 * np.pi * Fc * n / Fs)

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


def mov(y, window_len):
    y = np.asarray(y, dtype=float)
    N = y.shape[0]

    if window_len < 1 or window_len > N:
        raise ValueError("window_len must be between 1 and len(y)")

    # 1D averaging kernel
    kernel = np.ones(window_len) / window_len

    # Convolve with mode='same' → centered window (edges ramp)
    y_ma = np.convolve(y, kernel, mode="same")

    return y_ma


class CalibrationWorker(QThread):
    def run(self):
        wlbt.StartCalibration()
        stat, prog = wlbt.GetStatus()
        while stat == wlbt.STATUS_CALIBRATING and prog < 100:
            wlbt.Trigger()
            stat, prog = wlbt.GetStatus()
            print(f"Calibrating {prog}%")
            self.msleep(100)
        print("Calibration complete")


class ImageDisplayWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure, self.ax = plt.subplots(figsize=(5, 5))
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        #self.ax.axis('off')
        self.figure.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    def update_image(self, data):
        if data is not None:
            self.ax.clear()
            #self.ax.axis('off')
            self.figure.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
            self.ax.plot(data)
            
            self.canvas.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Radar Image Feed")

        # Layout setup
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        self.image_widget = ImageDisplayWidget()
        main_layout.addWidget(self.image_widget)

        controls_layout = QHBoxLayout()
        
        self.calibrate_button = QPushButton("Calibrate")
        self.calibrate_button.clicked.connect(self.calibrate_radar)
        controls_layout.addWidget(self.calibrate_button)

        self.reconnect_button = QPushButton("Reconnect")
        self.reconnect_button.clicked.connect(self.reconnect_radar)
        controls_layout.addWidget(self.reconnect_button)


        controls_layout.addStretch()

        self.status_label = QLabel("Radar Status: Disconnected")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        controls_layout.addWidget(self.status_label)
        main_layout.addLayout(controls_layout)

        self.radar_connected = False
        self.calibration_thread = None

        # ──── received signal matrix 's_l^rx(n, m)' containing (int) 'slow_time_N' past signals ────
        self.slow_time_N = 50
        self.s = deque(maxlen=self.slow_time_N)
        
        self.setup_radar()

        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_image)
        self.timer.start(100)
    
        self.apply_styles()

    def setup_radar(self):
        try:
            wlbt.Init()
            wlbt.Initialize()
            wlbt.ConnectAny()
            
            wlbt.SetProfile(wlbt.PROF_SENSOR)
            wlbt.SetDynamicImageFilter(wlbt.FILTER_TYPE_NONE)
    
            wlbt.Start()

            self.update_status(True)
        except Exception as e:
            print("Radar initialization failed:", e)
            self.update_status(False)

    # ──── This part gets the signal and performs calculations ────
    def refresh_image(self):
        try:
            wlbt.Trigger()
            
            # call getSignal()
            pairs = wlbt.GetAntennaPairs()
            x,  time = wlbt.GetSignal(pairs[0]) # i.e. pairs[0] is tx: 1 rx: 2
            x = np.array(x) 
            
            # 'x' is now equal to a single fast-time signal (1D array, dtype=double)
            
            # downconvert to baseband + downsample to number of frequency steps
            Fs = 102.4e9   # fast‐time sampling freq
            Fc = 7.15e9    # carrier freq
            B  = 1.7e9     # radar bandwidth
            y = downsample(x, Fs, Fc, B)
            
            # 'y' is now equal to a downconverted single fast-time signal (1D array, dtype=complex double)
            
            # save current signal to signal matrix
            self.s.append(y)
            
            # 's' is now a deque with the last 50 'y's
            
            # compute slow-time variance of 's'
            var = None
            if len(self.s) >= 2:
                # stack (deque) 's' to a numpy array
                arr = np.stack(self.s, axis=0)
                # variance along axis=0 (i.e. across 's')
                v = np.var(arr, axis=0, ddof=0)
                E = np.mean(arr)
                
                var = v / E
            
            # Update GUI
            self.image_widget.update_image(var)
        except Exception as e:
            print("Failed to retrieve radar data:", e)
            self.update_status(False)

    def calibrate_radar(self):
        if self.calibration_thread is None or not self.calibration_thread.isRunning():
            self.calibration_thread = CalibrationWorker()
            self.calibration_thread.start()

    def reconnect_radar(self):
        try:
            wlbt.Stop()
            wlbt.Disconnect()
        except Exception:
            pass  # Might already be disconnected
        
        wlbt.Clean()
        self.setup_radar()

    def update_status(self, connected: bool):
        self.radar_connected = connected
        if connected:
            self.status_label.setText("Radar Status: Connected")
            self.status_label.setProperty("status", "connected")
        else:
            self.status_label.setText("Radar Status: Disconnected")
            self.status_label.setProperty("status", "disconnected")
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def apply_styles(self):
        style = """
        QWidget {
            background-color: #1e1e1e;
            color: #dcdcdc;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 14px;
        }
        QPushButton {
            background-color: #0078d7;
            border: none;
            color: white;
            padding: 8px 16px;
            border-radius: 4px;
            font-weight: 600;
        }
        QPushButton:hover {
            background-color: #005a9e;
        }
        QPushButton:pressed {
            background-color: #003f6d;
        }
        QLabel {
            font-weight: 600;
        }
        QLabel[status="connected"] {
            color: #4caf50;
        }
        QLabel[status="disconnected"] {
            color: #f44336;
        }
        """
        self.setStyleSheet(style)

    def closeEvent(self, event):
        self.timer.stop()
        try:
            wlbt.Stop()
            wlbt.Disconnect()
        except Exception:
            pass
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(600, 600)
    window.show()
    sys.exit(app.exec())
