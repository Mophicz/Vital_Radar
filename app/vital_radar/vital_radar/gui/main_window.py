from collections import deque

from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import QTimer, Qt

from vital_radar.gui.widgets.image_display import ImageDisplayWidget
from vital_radar.walabot.connection import init_radar, stop_radar, reconnect_radar
from vital_radar.walabot.calibration import CalibrationWorker
from vital_radar.processing.raw_signal_processing import process_raw_signal
from vital_radar.processing.distance_estimation import slowVar


class MainWindow(QMainWindow):
    """
    Defines the main window for the radar GUI.
    
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Radar Image Feed")

        # background
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # plot/image area
        self.image_widget = ImageDisplayWidget()
        main_layout.addWidget(self.image_widget)

        # calibrate and reconnect buttons
        controls_layout = QHBoxLayout()
        self.calibrate_button = QPushButton("Calibrate")
        self.calibrate_button.clicked.connect(self.calibrate_radar)
        controls_layout.addWidget(self.calibrate_button)

        self.reconnect_button = QPushButton("Reconnect")
        self.reconnect_button.clicked.connect(self.reconnect_radar)
        controls_layout.addWidget(self.reconnect_button)

        # connection status label
        controls_layout.addStretch()
        self.status_label = QLabel("Radar Status: Disconnected")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        controls_layout.addWidget(self.status_label)
        main_layout.addLayout(controls_layout)

        self.radar_connected = False
        self.calibration_thread = None
        
        # signal matrix s
        self.slow_time_N = 50 # slow-time length
        self.s = deque(maxlen=self.slow_time_N) # use deque collection to keep matrix dimension constant (FIFO)

        # use walabot API to connect radar
        try:
            init_radar()
            self.update_status(True)
        except Exception as e:
            print("Radar initialization failed:", e)
            self.update_status(False)

        # refresh rate for GUI
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_image)
        self.timer.start(100) # ms

    def refresh_image(self):
        # if no radar is connected calling API functions will return an error, so skip
        if not self.radar_connected:
            return

        try:
            import WalabotAPI as wlbt
            
            # trigger radar to transmit and receive a signal
            wlbt.Trigger()
            pairs = wlbt.GetAntennaPairs()
            
            # get signals from the receive antennas
            x, _ = wlbt.GetSignal(pairs[0]) # example with one antenna pair
      
            # append signal to signal matrix (also removes oldest signal if full)
            self.s.append(process_raw_signal(x))
            
            # get slow-time variance
            var = slowVar(self.s)
            
            # update plot
            self.image_widget.update_image(var)
            
        except Exception as e:
            print("Failed to retrieve radar data:", e)
            self.update_status(False)

    def calibrate_radar(self):
        # if no radar is connected calling API functions will return an error, so skip
        if not self.radar_connected:
            return

        if self.calibration_thread is None or not self.calibration_thread.isRunning():
            self.calibration_thread = CalibrationWorker()
            self.calibration_thread.start()

    def reconnect_radar(self):
        try:
            success = reconnect_radar()
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

    def closeEvent(self, event):
        self.timer.stop()
        stop_radar()
        event.accept()
        