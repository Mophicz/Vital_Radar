from collections import deque

from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFontMetrics, QFont

from vital_radar.gui.widgets.image_display import ImageDisplayWidget
from vital_radar.walabot.connection import initRadar, stopRadar, reconnectRadar
from vital_radar.walabot.calibration import CalibrationWorker
import vital_radar.walabot.signal_aquisition as sa
from vital_radar.processing.display_modes import DisplayMode, computePlotData
from vital_radar.processing.raw_signal_processing import processRawSignal
from vital_radar.processing.utils import getStack


class MainWindow(QMainWindow):
    """Defines the main window for the radar GUI."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vital Radar")

        # default display mode
        self.current_display_mode = DisplayMode.RAW

        # central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QVBoxLayout(central_widget)

        # plot area
        self.image_widget = self._build_plot_area()
        self.main_layout.addWidget(self.image_widget)

        # radar control area
        self.status_label = QLabel("Radar Status: Disconnected")
        
        mono = QFont("Courier New")
        fm = QFontMetrics(mono)
        w = fm.horizontalAdvance("00.0")
        
        self.freq_value = QLabel("00.0")
        self.freq_value.setFont(mono)
        self.freq_value.setFixedWidth(w)
       
        self.distance_label = QLabel("Distance:")
        
        self.distance_value = QLabel("00.0")
        self.distance_value.setFont(mono)
        self.distance_value.setFixedWidth(w)
        
        self.distance_label.setVisible(False)
        self.distance_value.setVisible(False)
        self.main_layout.addWidget(self._build_control_area())

        # radar and buffer
        self.radar_connected = False
        self.slow_time_N = 50
        self.signal_buffer = deque(maxlen=self.slow_time_N)

        # Initialize radar
        try:
            initRadar()
            self.updateStatus(True)
        except Exception as e:
            print("Radar initialization failed:", e)
            self.updateStatus(False)

        # GUI refresh timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.refreshImage)
        self.timer.start(100)

    def _build_plot_area(self):
        """Returns the widget containing the radar image."""
        image_w = ImageDisplayWidget()
        return image_w
    
    def _build_control_area(self):
        """Returns a QWidget with all buttons and labels laid out."""
        container = QWidget()
        hbox = QHBoxLayout(container)
        hbox.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        # Buttons
        hbox.addWidget(self._build_button("Calibrate", self.calibrateRadar))
        hbox.addWidget(self._build_button("Reconnect", self.reconnectRadar))

        hbox.addStretch()
        
        # Dropdown (centered)
        self.mode_combo = QComboBox()
        for mode in DisplayMode:
            self.mode_combo.addItem(mode.name, mode)
        self.mode_combo.setCurrentText(self.current_display_mode.name)
        self.mode_combo.currentIndexChanged.connect(self.modeChanged)
        hbox.addWidget(self.mode_combo, alignment=Qt.AlignmentFlag.AlignCenter)

        hbox.addWidget(self.distance_label, alignment=Qt.AlignmentFlag.AlignCenter)
        hbox.addWidget(self.distance_value, alignment=Qt.AlignmentFlag.AlignCenter)
        
        hbox.addStretch()

        # Status labels
        vbox = QVBoxLayout()
        vbox.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
        vbox.addWidget(self.status_label)
        
        freq_container = QWidget()
        freq_layout = QHBoxLayout(freq_container)
        freq_layout.addWidget(QLabel("Trigger Frequency:"))
        freq_layout.addWidget(self.freq_value)
        vbox.addWidget(freq_container)
        hbox.addLayout(vbox)

        return container
    
    def _build_button(self, text, slot):
        btn = QPushButton(text)
        btn.clicked.connect(slot)
        return btn
    
    def refreshImage(self):
        # if no radar is connected calling API functions will return an error, so skip
        if not self.radar_connected:
            return

        # choose transmit and receive antennas
        desired_tx = [1, 17]
        desired_rx = [2, 6, 10, 14]

        # get the corresponding signals
        signals = sa.getSignals(desired_tx, desired_rx)
        
        # process the raw data to downsampled I/Q signals
        iq_signals = processRawSignal(signals)
        
        # append signal to signal matrix (deque also removes oldest signal if full)
        self.signal_buffer.append(iq_signals)
        
        signal_matrix = getStack(self.signal_buffer)
        
        data, distance = computePlotData(signal_matrix, self.current_display_mode)
        
        if distance is not None:
            self.distance_value.setText(f"{distance:04.1f}")
        else:
            self.distance_value.setText("00.0")
            
        self.freq_value.setText(f"{sa.trigger_freq:04.1f}")
        
        # update plot
        self.image_widget.updateImage(data)

    def calibrateRadar(self):
        # if no radar is connected calling API functions will return an error, so skip
        if not self.radar_connected:
            return

        if self.calibration_thread is None or not self.calibration_thread.isRunning():
            self.calibration_thread = CalibrationWorker()
            self.calibration_thread.start()

    def reconnectRadar(self):
        try:
            success = reconnectRadar()
            self.updateStatus(success)
        except Exception as e:
            print("Reconnect failed:", e)
            self.updateStatus(False)

    def modeChanged(self, index):
        self.current_display_mode = self.mode_combo.currentData()
        if self.current_display_mode == DisplayMode.DISTANCE:
            self.distance_label.setVisible(True)
            self.distance_value.setVisible(True)
        else:
            self.distance_label.setVisible(False)
            self.distance_value.setVisible(False)
        
    def updateStatus(self, connected: bool):
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
        stopRadar()
        event.accept()
        