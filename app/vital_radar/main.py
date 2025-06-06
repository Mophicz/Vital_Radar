import sys, os
from PyQt6.QtWidgets import QApplication
from PyQt6 import QtGui
from vital_radar.gui.main_window import MainWindow

if getattr(sys, "frozen", False):
    basedir = sys._MEIPASS
else:
    basedir = os.path.dirname(__file__)

ICON_PATH = os.path.join(basedir, "vital_radar", "gui", "resources", "vital_radar_icon.ico")

def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon(ICON_PATH))
    window = MainWindow()
    window.resize(600, 600)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()