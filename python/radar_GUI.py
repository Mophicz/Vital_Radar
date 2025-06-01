import sys
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
        self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)

    def update_image(self, image_array, marker_coords):
        self.ax.clear()
        #self.ax.axis('off')
        self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.ax.imshow(image_array, cmap='viridis', aspect='auto')
        
        if marker_coords is not None:
            y_idx, z_idx = marker_coords
            self.ax.plot(y_idx, z_idx, 'r+', markersize=12, markeredgewidth=2)  # red cross
        
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
            wlbt.SetDynamicImageFilter(wlbt.FILTER_TYPE_DERIVATIVE)
            wlbt.SetThreshold(35)
            
            wlbt.SetArenaR(1, 100, 2)
            wlbt.SetArenaTheta(-20, 20, 10)
            wlbt.SetArenaPhi(-45, 45, 2)
            wlbt.Start()

            self.update_status(True)
        except Exception as e:
            print("Radar initialization failed:", e)
            self.update_status(False)

    def refresh_image(self):
        if not self.radar_connected:
            self.image_widget.update_image(np.zeros((100, 100)))
            return

        try:
            wlbt.Trigger()
            
            targets = wlbt.GetSensorTargets()
            print(f"Targets: {targets}")
        
            raw_image, _, _, _, _ = wlbt.GetRawImage()
            raw_image = np.array(raw_image)  # shape (theta, phi, r)
            shape = raw_image.shape
            
            r_min, r_max, r_res = wlbt.GetArenaR()
            phi_min, phi_max, phi_res = wlbt.GetArenaPhi()
            theta_min, theta_max, theta_res = wlbt.GetArenaTheta()
            
            r = np.linspace(r_min, r_max, shape[2])
            phi = np.linspace(phi_min, phi_max, shape[1])
            theta = np.linspace(theta_min, theta_max, shape[0])
            
            # Interpolator setup
            interp = RegularGridInterpolator((theta, phi, r), raw_image, bounds_error=False, fill_value=0)

            # Define a Cartesian grid (central slice in YZ plane, i.e., x ≈ 0)
            num_points = 100
            y = np.linspace(-r_max, r_max, num_points)
            z = np.linspace(0, r_max, num_points)
            Y, Z = np.meshgrid(y, z)
            X = np.zeros_like(Y)

            # Convert to spherical coordinates
            r_cart = np.sqrt(X**2 + Y**2 + Z**2)
            theta_cart = np.arccos(np.divide(X, r_cart, where=r_cart!=0))  # avoid divide by zero
            phi_cart = np.arctan2(Y, Z)

            # Stack for interpolation
            coords = np.stack([theta_cart, phi_cart, r_cart], axis=-1)

            # Interpolate
            central_slice = interp(coords)

            marker_coords = None
            if targets:
                target = targets[0]
                y_target, z_target = target.yPosCm, target.zPosCm

                y_idx = int((y_target - y[0]) / (y[-1] - y[0]) * (num_points - 1))
                z_idx = int((z_target - z[0]) / (z[-1] - z[0]) * (num_points - 1))

                y_idx = np.clip(y_idx, 0, num_points - 1)
                z_idx = np.clip(z_idx, 0, num_points - 1)

                marker_coords = (y_idx, z_idx)
                
            # Update GUI
            self.image_widget.update_image(central_slice, marker_coords)
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
