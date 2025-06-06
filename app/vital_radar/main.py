import sys
import os
from PyQt6 import QtWidgets, QtGui

if getattr(sys, "frozen", False):
    basedir = sys._MEIPASS
else:
    basedir = os.path.dirname(__file__)

ICON_PATH = os.path.join(basedir, "vital_radar", "gui", "resources", "vital_radar_icon.ico")

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vital Radar")
        label = QtWidgets.QLabel("Vital Radar App")
        label.setMargin(10)
        self.setCentralWidget(label)
        self.show()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon(ICON_PATH))
    w = MainWindow()
    app.exec()