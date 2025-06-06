from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import QTimer, Qt
from collections import deque
import numpy as np

from vital_radar.gui.widgets.image_display import ImageDisplayWidget
from vital_radar.walabot.connection import init_radar, stop_radar, reconnect_radar as wb_reconnect
from vital_radar.walabot.calibration import CalibrationWorker
from vital_radar.processing.downsampling import downsample

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Radar Image Feed")

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
        self.slow_time_N = 50
        self.s = deque(maxlen=self.slow_time_N)

        # Instead of calling setup_radar directly here, call init_radar()
        try:
            init_radar()
            self.update_status(True)
        except Exception as e:
            print("Radar initialization failed:", e)
            self.update_status(False)

        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_image)
        self.timer.start(100)

        self.apply_styles()

    def refresh_image(self):
        if not self.radar_connected:
            return

        try:
            import WalabotAPI as wlbt  # still need to fetch frames
            wlbt.Trigger()
            pairs = wlbt.GetAntennaPairs()
            x, _ = wlbt.GetSignal(pairs[0])
            x = np.array(x)
            Fs = 102.4e9
            Fc = 7.15e9
            B = 1.7e9
            y = downsample(x, Fs, Fc, B)
            self.s.append(y)
            var = None
            if len(self.s) >= 2:
                arr = np.stack(self.s, axis=0)
                v = np.var(arr, axis=0, ddof=0)
                E = np.mean(arr)
                var = v / E
            self.image_widget.update_image(var)
        except Exception as e:
            print("Failed to retrieve radar data:", e)
            self.update_status(False)

    def calibrate_radar(self):
        if not self.radar_connected:
            return

        if self.calibration_thread is None or not self.calibration_thread.isRunning():
            self.calibration_thread = CalibrationWorker()
            self.calibration_thread.start()

    def reconnect_radar(self):
        """
        Call the helper that does stop + init underneath.
        """
        try:
            success = wb_reconnect()
            self.update_status(success)
        except Exception as e:
            print("Reconnect failed:", e)
            self.update_status(False)

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
        stop_radar()
        event.accept()
