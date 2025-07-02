from collections import deque

from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont, QFontMetrics
import numpy as np

from vital_radar.gui.widgets.image_display import ImageDisplayWidget
from vital_radar.gui.widgets.antenna_matrix import AntennaMatrix, tx_to_rx
from vital_radar.walabot.connection import initRadar, stopRadar, reconnectRadar
from vital_radar.walabot.calibration import CalibrationWorker
import vital_radar.walabot.signal_aquisition as sa
from vital_radar.processing.display_modes import DisplayMode, computePlotData
from vital_radar.processing.raw_signal_processing import processRawSignal, downsample_raw
from vital_radar.processing.utils import getStack, dummy_signal_generator
        

class MainWindow(QMainWindow):
    """
    Defines the main window for the radar GUI.
    
    """
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
        self.image_widget = self._buildPlotArea()
        self.main_layout.addWidget(self.image_widget)

        # status label
        self.status_label = QLabel("Radar Status: Disconnected")
        
        # frequency label
        self.freq_value = QLabel("00.0")
        self.freq_value.setFont(QFont("Courier New"))
        
        # radar control area
        self.main_layout.addWidget(self._buildControlArea())

        # antenna matrix area
        self.matrix = AntennaMatrix(tx_to_rx)
        self.main_layout.addWidget(self.matrix, alignment=Qt.AlignmentFlag.AlignCenter)
        self.matrix.selectionChanged.connect(self.onMatrixChange)
        self.selected_pairs: set[tuple[int,int]] = set()
        
        defaults = [(1,2), (1,6), (1,10), (1,14)]
        self.matrix.apply_defaults(defaults)
        
        # radar and buffer
        self.radar_connected = False
        self.slow_time_N = 100
        self.signal_buffer = deque(maxlen=10)
        self.avg_signal_buffer = deque(maxlen=self.slow_time_N)
        
        self.dummy_signal_generator = dummy_signal_generator()

        # Initialize radar
        try:
            initRadar()
            self.updateStatus(True)
        except Exception as e:
            print("Radar initialization failed:", e)
            self.updateStatus(False)

        self.calibration_thread = None
        
        # GUI refresh timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.refreshImage)
        self.timer.start(100)
    
    def refreshImage(self):
        """
        This function is called repeatedtly as long as the GUI is running and updates the displayed image.
        """  
            
        if not self.selected_pairs:
            return
        
        # if no radar is connected use dummy data instead
        if not self.radar_connected:
            signals = next(self.dummy_signal_generator)
            self.freq_value.setText(f"--.-")
        else:
            # get the signals from the walabot API
            signals = sa.getSignals(self.selected_pairs)
            self.freq_value.setText(f"{sa.trigger_freq:04.1f}")

        # for RAW dislplay mode the signals arent processed
        if self.current_display_mode == DisplayMode.RAW:
            signals = downsample_raw(signals, 10)
        else:
            # for all other modes the IQ signals are used
            signals = processRawSignal(signals)
        
        # append signals to signal buffer
        self.signal_buffer.append(signals)

        avg_signal = np.mean(self.signal_buffer, axis=0)
        self.avg_signal_buffer.append(avg_signal)
        
        # convert buffer to matrix
        signal_matrix = getStack(self.avg_signal_buffer)

        # compute plot data
        plot_data = computePlotData(signal_matrix, self.current_display_mode, self.selected_pairs)

        # update plot
        self.image_widget.updateImage(plot_data, self.current_display_mode)
        
    def calibrateRadar(self):
        """
        Slot connected to calibration button.
        
        """
        # if no radar is connected calling API functions will return an error, so skip
        if not self.radar_connected:
            return

        if self.calibration_thread is None or not self.calibration_thread.isRunning():
            self.calibration_thread = CalibrationWorker()
            self.calibration_thread.start()
            
        self.signal_buffer.clear()
        self.avg_signal_buffer.clear()

    def reconnectRadar(self):
        """
        Slot connected to reconnect button.
        
        """
        try:
            success = reconnectRadar()
            self.updateStatus(success)
        except Exception as e:
            print("Reconnect failed:", e)
            self.updateStatus(False)
        
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
        """
        Stop event loop and radar when window is closed.
        
        """
        self.timer.stop()
        stopRadar()
        event.accept()
    
    def modeChanged(self):
        """
        Slot connected to mode selection dropdown menu.
        
        """
        self.current_display_mode = self.mode_combo.currentData()
        self.signal_buffer.clear()
        self.avg_signal_buffer.clear()
        self.image_widget.clear(self.current_display_mode)
        
    def onMatrixChange(self, tx: int, rx: int, checked: bool):
        """
        Connected to checkbox check.
        
        """
        if checked:
            self.selected_pairs.add((tx, rx))
        else:
            self.selected_pairs.discard((tx, rx))
            
        self.signal_buffer.clear()
        self.avg_signal_buffer.clear()
        self.image_widget.clear(self.current_display_mode)
    
    def _buildPlotArea(self):
        """
        Returns the widget containing the radar image.
        
        """
        image_w = ImageDisplayWidget()
        return image_w
    
    def _buildControlArea(self):
        """
        Returns a QWidget with all buttons and labels laid out.
        
        """
        container = QWidget()
        hbox = QHBoxLayout(container)
        hbox.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        # buttons (left)
        hbox.addWidget(self._buildButton("Calibrate", self.calibrateRadar))
        hbox.addWidget(self._buildButton("Reconnect", self.reconnectRadar))

        # add gap 
        hbox.addStretch()
        
        # dropdown (center)
        self.mode_combo = QComboBox()
        for mode in DisplayMode:
            self.mode_combo.addItem(mode.name, mode)
        self.mode_combo.setCurrentText(self.current_display_mode.name)
        self.mode_combo.currentIndexChanged.connect(self.modeChanged)
        
        # Get font metrics for current font
        metrics = QFontMetrics(self.mode_combo.font())

        # Measure all item text widths
        max_text_width = max(metrics.horizontalAdvance(self.mode_combo.itemText(i)) for i in range(self.mode_combo.count()))

        # Add some padding (icon space, dropdown arrow, margin)
        padding = 40  # tweak this if needed

        # Apply the final width
        self.mode_combo.setMinimumWidth(max_text_width + padding)
        
        hbox.addWidget(self.mode_combo, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # add gap 
        hbox.addStretch()

        # labels
        vbox = QVBoxLayout()
        vbox.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
        vbox.addWidget(self.status_label, alignment=Qt.AlignmentFlag.AlignRight)
        
        freq_layout = QHBoxLayout()
        freq_layout.addWidget(QLabel("Trigger Frequency: "))
        freq_layout.addWidget(self.freq_value)
        freq_layout.addWidget(QLabel("Hz"))
        vbox.addLayout(freq_layout)
        hbox.addLayout(vbox)

        return container
    
    def _buildButton(self, text, slot):
        """
        Returns a button widget and connects it to a slot.
        
        """
        btn = QPushButton(text)
        btn.clicked.connect(slot)
        return btn
    