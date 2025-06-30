from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np

from vital_radar.processing.distance_estimation import sample2range
from vital_radar.processing.display_modes import DisplayMode
from vital_radar.processing.utils import moving_average
from vital_radar.processing.spectrum_estimation import getWelch, getARpsd, bandpassFilter
import vital_radar.walabot.signal_aquisition as sa


class ImageDisplayWidget(QWidget):
    """
    Defines the widget for displaying plots of radar data with matplotlib.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create a matplotlib figure and a single axis by default
        self.figure, self.ax = plt.subplots(figsize=(5, 5), constrained_layout=True)

        # Embed the figure into the Qt widget
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        self.canvas.updateGeometry()

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def updateImage(self, data, display_mode):
        # Clear the figure to prepare for new plots
        self.figure.clear()
        
        if data is None:
            data = np.zeros(100)
        
        match display_mode:
            case DisplayMode.RAW:
                ax = self.figure.add_subplot(1, 1, 1)
                self._plotRaw(ax, data)
                self.ax = ax
            case DisplayMode.IQ:
                ax = self.figure.add_subplot(1, 1, 1)
                self._plotIQ(ax, data)
                self.ax = ax
            case DisplayMode.DISTANCE:
                ax = self.figure.add_subplot(1, 1, 1)
                self._plotDistance(ax, data)
                self.ax = ax
            case DisplayMode.BREATHING:
                # Create two subplots side by side
                ax_time = self.figure.add_subplot(1, 2, 1)
                ax_psd = self.figure.add_subplot(1, 2, 2)
                self._plotBreathing(ax_time, ax_psd, data)
                self.ax = ax_time  # store a reference if needed

        self.canvas.draw()

    def clear(self, display_mode):
        self.updateImage(None, display_mode)
        
    def _plotRaw(self, ax, data):
        N = data.shape[0]
        x = np.arange(N)
        end = int(np.ceil(x.max()))
        ax.plot(x, data)
        ax.set_xlabel('Sample index (k)')
        ax.set_xticks(np.arange(0, end, 100))
        ax.set_xlim(0, end)
        ax.set_ylabel('Signal amplitude')
        ax.set_ylim(-1, 1)

    def _plotIQ(self, ax, data):
        N = data.shape[0]
        x = sample2range(np.arange(N))
        end = int(np.ceil(x.max()))
        real_part = np.real(data)
        imag_part = np.imag(data)
        ax.plot(x, real_part, label='Real')
        ax.plot(x, imag_part, label='Imaginary')
        ax.set_xlabel('Range (m)')
        ax.set_xticks(np.arange(end))
        ax.set_xlim(0, end)
        ax.set_ylabel('Signal amplitude')
        ax.set_ylim(-1, 1)
        ax.legend(loc='upper right')

    def _plotDistance(self, ax, data):
        N = data.shape[0]
        x = sample2range(np.arange(N))
        end = int(np.ceil(x.max()))
        peak_idx = np.argmax(data)
        peak_range = x[peak_idx]
        ax.plot(x, np.abs(data)**2)
        ax.set_xlabel('Range (m)')
        ticks = [0, peak_range, end]
        labels = ['0', f'{peak_range:.2f}', str(end)]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_xlim(0, end)
        ax.set_ylabel('Normalized slow time variance')
        #ax.set_ylim(0, 1)
        ax.axvline(peak_range, color='red', linestyle='--', label='Peak distance')
        for tick_val, tick_label in zip(ax.get_xticks(), ax.get_xticklabels()):
            if np.isclose(tick_val, peak_range):
                tick_label.set_color('red')
        ax.legend()

    def _plotBreathing(self, ax_time, ax_psd, data):
        fs = sa.trigger_freq
        
        # handle cases when trigger_freq is NaN
        if not np.isfinite(fs):
            fs = 1
            
        if len(data) < 9:
            return
        
        # remove DC content
        y = data - np.mean(data)
        
        # apply moving average
        y_smooth = moving_average(y, 20)

        # FFT & PSD
        f, P = getWelch(y_smooth, fs)
        #f, P = getARpsd(y_smooth, fs)

        k = np.arange(-len(y_smooth), 0, 1)

        # Plot time-domain signal
        ax_time.plot(k, y_smooth)
        
        ax_time.set_title('Beamformed Slow-Time Signal')
        
        ax_time.set_xlabel('Sample (k)')
        
        ax_time.set_ylabel('Amplitude')

        # Plot frequency-domain PSD (normalized)
        ax_psd.semilogy(f, P)
        
        ax_psd.set_title('PSD')
        
        ax_psd.set_xlabel('Frequency (Hz)')
        ax_psd.set_xlim(-0.05, 1.55)
        
        ax_psd.set_ylabel('Logarithmic PSD')

        ax_psd.axvline(0.2, color='red', linestyle='--', label='Expected Breathing Range')
        ax_psd.axvline(0.3, color='red', linestyle='--')