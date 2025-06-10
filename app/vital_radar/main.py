import sys, os

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer, Qt, QFile, QTextStream
from PyQt6 import QtGui

from vital_radar.gui.main_window import MainWindow

if getattr(sys, "frozen", False):
    basedir = sys._MEIPASS
else:
    basedir = os.path.dirname(__file__)

ICON_PATH = os.path.join(basedir, "vital_radar", "gui", "resources", "vital_radar_icon.ico")
STYLE_PATH = os.path.join(basedir, "vital_radar", "gui", "resources", "style.qss")

def load_stylesheet(app: QApplication):
    file = QFile(STYLE_PATH)
    if not file.open(QFile.OpenModeFlag.ReadOnly | QFile.OpenModeFlag.Text):
        print(f"Could not load stylesheet at: {STYLE_PATH}")
        return
    stream = QTextStream(file)
    app.setStyleSheet(stream.readAll())
    file.close()   
    
def main():
    app = QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon(ICON_PATH))
    load_stylesheet(app)
    window = MainWindow()
    window.resize(600, 600)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()