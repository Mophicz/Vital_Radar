import sys, os

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QFile, QTextStream
from PyQt6 import QtGui

from vital_radar.gui.main_window import MainWindow


# get base directory of this script
if getattr(sys, "frozen", False):
    basedir = sys._MEIPASS
else:
    basedir = os.path.dirname(__file__)
    
# build paths to GUI resources
ICON_PATH = os.path.join(basedir, "vital_radar", "gui", "resources", "vital_radar_icon.ico")
STYLE_PATH = os.path.join(basedir, "vital_radar", "gui", "resources", "style.qss")


def load_stylesheet(app: QApplication):
    """
    Loads a file with the style formatting for the GUI windows and applies it to a given QApplication.
    
    """
    file = QFile(STYLE_PATH)
    if not file.open(QFile.OpenModeFlag.ReadOnly | QFile.OpenModeFlag.Text):
        print(f"Could not load stylesheet at: {STYLE_PATH}")
        return
    stream = QTextStream(file)
    app.setStyleSheet(stream.readAll())
    file.close()   
  
    
def main():
    """
    Main function to start the GUI action loop.
    
    """
    # create QApplication
    app = QApplication(sys.argv)
    
    # set the custom icon 
    app.setWindowIcon(QtGui.QIcon(ICON_PATH))
    
    # load the custom styles
    load_stylesheet(app)
    
    # create an instance of MainWindow()
    window = MainWindow()
    
    # set custom window dimensions
    window.resize(600, 600)
    
    # open the window
    window.show()
    
    # end the code when the app is closed
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
    