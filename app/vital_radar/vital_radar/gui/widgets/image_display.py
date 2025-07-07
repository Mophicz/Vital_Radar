from collections import deque

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from vital_radar.processing.distance_estimation import sample2range
from vital_radar.processing.display_modes import DisplayMode
from vital_radar.processing.utils import moving_average
from vital_radar.processing.spectrum_estimation import getWelch, getARpsd, bandpassFilter
import vital_radar.walabot.signal_aquisition as sa


class ImageDisplayWidget(QWidget):
    """
    Defines the widget for displaying plots with matplotlib.
    
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # create a matplotlib figure
        self.figure, self.ax = plt.subplots(figsize=(5, 5), constrained_layout=True)

        # embed the figure into the Qt widget
        self.canvas = FigureCanvas(self.figure)
        
        # automatically fill the available space
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        self.canvas.updateGeometry()

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        self.buffer = deque(maxlen=50)

    def updateImage(self, data, display_mode):
        # clear the figure to prepare for new plots
        self.figure.clear()
        
        # default if no data is passed
        if data is None:
            data = np.zeros(100)
        
        match display_mode:
            case DisplayMode.RAW:
                # single plot
                ax = self.figure.add_subplot(1, 1, 1)
                self._plotRaw(ax, data)
                self.ax = ax
            case DisplayMode.IQ:
                #single plot
                ax = self.figure.add_subplot(1, 1, 1)
                self._plotIQ(ax, data)
                self.ax = ax
            case DisplayMode.DISTANCE:
                #single plot
                ax = self.figure.add_subplot(1, 1, 1)
                self._plotDistance(ax, data)
                self.ax = ax
            case DisplayMode.BREATHING:
                # two subplots side by side
                ax_time = self.figure.add_subplot(1, 2, 1)
                ax_psd = self.figure.add_subplot(1, 2, 2)
                self._plotBreathing(ax_time, ax_psd, data)

        self.canvas.draw()

    def clear(self, display_mode):
        self.updateImage(None, display_mode)
        
    def _plotRaw(self, ax, data):
        # x-axis array
        N = data.shape[0]
        x = np.arange(N)
        
        # plot
        ax.plot(x, data)
        
        # x-axis customization
        ax.set_xlabel('Sample index (k)')
        # round up max value
        end = int(np.ceil(x.max()))
        # ticks
        ax.set_xticks(np.arange(0, end, 100))
        # limits
        ax.set_xlim(0, end)
        
        # y-axis customization
        ax.set_ylabel('Signal amplitude')
        # custom limits (data normed by max)
        ax.set_ylim(-1, 1)

    def _plotIQ(self, ax, data):
        # x-axis array
        N = data.shape[0]
        x = np.arange(N)
        
        # real and imaginary parts
        real_part = np.real(data)
        imag_part = np.imag(data)
        
        # plot
        ax.plot(x, real_part, label='Real')
        ax.plot(x, imag_part, label='Imaginary')
        
        # x-axis customization
        ax.set_xlabel('Sample index (k)')
        # round up max value
        end = int(np.ceil(x.max()))
        # ticks
        ax.set_xticks(np.arange(end))
        # limits
        ax.set_xlim(0, end)
        
        # y-axis customization
        ax.set_ylabel('Signal amplitude')
        # limits
        ax.set_ylim(-1, 1)
        
        # legend
        ax.legend(loc='upper right')

    def _plotDistance(self, ax, data):
        # x-axis array
        N = data.shape[0]
        x = sample2range(np.arange(N))
        
        # find peak index
        peak_idx = np.argmax(data)
        # convert to range
        peak_range = x[peak_idx]
        
        # plot
        ax.plot(x, np.abs(data)**2)
        
        # x-axis customization
        ax.set_xlabel('Range (m)')
        # round up max value
        end = int(np.ceil(x.max()))
        # ticks
        ticks = [0, peak_range, end]
        ax.set_xticks(ticks)
        # custom labels for round numbers
        labels = ['0', f'{peak_range:.2f}', str(end)]
        ax.set_xticklabels(labels)
        # limits
        ax.set_xlim(0, end)
        
        # y-axis customization
        ax.set_ylabel('Normalized slow time variance')
        
        # red line to highlight maximum
        ax.axvline(peak_range, color='red', linestyle='--', label='Peak distance')
        for tick_val, tick_label in zip(ax.get_xticks(), ax.get_xticklabels()):
            if np.isclose(tick_val, peak_range):
                tick_label.set_color('red')
            
        # legend
        ax.legend()

    def _plotBreathing(self, ax_time, ax_psd, data):
        fs = sa.trigger_freq
        
        # handle cases when trigger_freq is NaN
        if not np.isfinite(fs):
            fs = 1
            
        if len(data) < 10:
            return
        
        x = moving_average(data, 30)
        
        fc = 0.1
        b, a = signal.butter(2, fc/(fs/2), btype='high')
        x = signal.filtfilt(b, a, x)
        
        fc = 0.6
        b, a = signal.butter(2, fc/(fs/2), btype='low')
        x = signal.filtfilt(b, a, x)
        
        self.buffer.append(x[-1])
        buffer = np.array(self.buffer)
        
        # FFT & PSD
        f, P = getWelch(x, fs)

        k = np.arange(-len(buffer), 0, 1)

        # Plot time-domain signal
        ax_time.plot(k/np.ceil(fs), buffer)
        
        ax_time.set_title('Time Signal')
        
        ax_time.set_xlabel('Time (s)')
        
        ax_time.set_ylabel('Amplitude')
        
        ax_time.set_ylim(-0.005, 0.005)

        # Plot frequency-domain PSD (normalized)
        ax_psd.plot(f, P)
        
        ax_psd.set_title('Spectrum Estimate')
        
        ax_psd.set_xlabel('Frequency (Hz)')
        ax_psd.set_xlim(0, 1)
        
        ax_psd.set_ylabel('Logarithmic PSD')

        ax_psd.axvline(0.2, color='red', linestyle='--', label='Expected Breathing Range')
        ax_psd.axvline(0.3, color='red', linestyle='--')
        
        ax_psd.legend()
        