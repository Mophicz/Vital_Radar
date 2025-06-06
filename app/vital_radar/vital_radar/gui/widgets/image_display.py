from PyQt6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

class ImageDisplayWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure, self.ax = plt.subplots(figsize=(5, 5))
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.figure.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    def update_image(self, data):
        if data is not None:
            self.ax.clear()
            self.figure.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
            self.ax.plot(data)
            self.canvas.draw()