from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np

from vital_radar.processing.distance_estimation import sample2range
from vital_radar.processing.display_modes import DisplayMode


class ImageDisplayWidget(QWidget):
    """
    Defines the widget for displaying plots of radar data with matplotlib.
    
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.figure, self.ax = plt.subplots(figsize=(5,5), constrained_layout=True)

        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        self.canvas.updateGeometry()

        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)  
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def updateImage(self, data, display_mode):
        self.ax.clear()
        
        if data is None:   
            data = np.zeros(137)
         
        match display_mode:
            case DisplayMode.RAW:
                self._plotRaw(data)
                
            case DisplayMode.IQ:
                self._plotIQ(data)
                
            case DisplayMode.DISTANCE:
                self._plotDistance(data)
                
        self.canvas.draw()
            
    def _plotRaw(self, data):
        N = data.shape[0]
        x = np.arange(N)
        
        end = int(np.ceil(x.max()))
        
        self.ax.plot(x, data)
        
        self.ax.set_xlabel('Sample index (k)')
        self.ax.set_xticks(np.arange(0, end, 100))
        self.ax.set_xlim(0, end)
        
        self.ax.set_ylabel('Signal amplitude')
        self.ax.set_ylim(-1, 1)
        
    def _plotIQ(self, data):
        N = data.shape[0]
        x = sample2range(np.arange(N))

        end = int(np.ceil(x.max()))
        
        real_part = np.real(data)
        imag_part = np.imag(data)

        self.ax.plot(x, real_part, label='Real', color='#1f77b4')
        self.ax.plot(x, imag_part, label='Imaginary', color='#FF8C00')
        
        self.ax.set_xlabel('Range (m)')
        self.ax.set_xticks(np.arange(end))
        self.ax.set_xlim(0, end)
        
        self.ax.set_ylabel('Signal amplitude')
        self.ax.set_ylim(-1, 1)
        
        self.ax.legend(['In phase', 'Quadrature'], loc='upper right')      
    
    def _plotDistance(self, data):
        N = data.shape[0]
        x = sample2range(np.arange(N))
        
        end = int(np.ceil(x.max()))
        
        # Find the peak index and its range
        peak_idx   = np.argmax(data)
        peak_range = x[peak_idx]
        
        # Plot the data
        self.ax.plot(x, data)
        
        self.ax.set_xlabel('Range (m)')
        ticks  = [0, peak_range, end]
        labels = ['0', f'{peak_range:.2f}', str(end)]
        self.ax.set_xticks(ticks)
        self.ax.set_xticklabels(labels)
        self.ax.set_xlim(0, end)

        self.ax.set_ylabel('Normalized slow time variance')
        self.ax.set_ylim(-1, 1)
        
        # Draw a vertical red dashed line at the peak
        self.ax.axvline(peak_range, color='red', linestyle='--', label='Peak distance')
        
        # Color the peak tick label red
        for tick_val, tick_label in zip(self.ax.get_xticks(), self.ax.get_xticklabels()):
            if np.isclose(tick_val, peak_range):
                tick_label.set_color('red')
                