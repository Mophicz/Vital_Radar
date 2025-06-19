from PyQt6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np

from vital_radar.processing.distance_estimation import sample2range
from vital_radar.processing.display_modes import DisplayMode


class ImageDisplayWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure, self.ax = plt.subplots(figsize=(5, 5))
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        # add padding to axis are visble
        self.figure.subplots_adjust(left=0.06, right=0.96, top=0.93, bottom=0.12)

    def updateImage(self, data, display_mode):
        self.ax.clear()
        
        if data is None:   
            return  
         
        if display_mode == DisplayMode.RAW:
            return
        elif display_mode == DisplayMode.VARIANCE:
            N = data.shape[0]
            x = sample2range(np.arange(N))
            
            self.ax.plot(x, data)
            self.ax.set_xticks(np.arange(int(np.ceil(x.max()+1))))
            
        self.canvas.draw()
            