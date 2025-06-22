import numpy as np
    
    
from scipy.constants import c


class DelaySumBeamformer:
    """
    A simple delay-and-sum beamformer assuming signal_matrix is (slow-time, fast-time, channels).

    Parameters
    ----------
    element_positions : (L, 3) array
        3D coordinates of each radar element (monostatic assumption) or virtual channel.
    frequencies : (F,) array
        Fast-time frequency vector corresponding to signal_matrix axis 1 (in Hz).
    """
    def __init__(self, element_positions, frequencies):
        self.positions = np.asarray(element_positions)  # (L, 3)
        self.freqs = np.asarray(frequencies)            # (F,)
        # Precompute angular frequencies
        self.omega = 2 * np.pi * self.freqs            # (F,)

    def _compute_delays(self, target_point):
        """
        Compute one-way delays from each element to the target and back

        tau_i = 2 * ||pos_i - r|| / c

        Parameters
        ----------
        target_point : (3,) array
            3D coordinates of the focus point.

        Returns
        -------
        delays : (L,) array of round-trip delays
        """
        # distances: shape (L,)
        d = np.linalg.norm(self.positions - target_point[None, :], axis=1)
        return 2 * d / c  # shape (L,)

    def _compute_weights(self, delays):
        """
        Steering weights per element and frequency

        w_i(f) = exp(-j * omega(f) * tau_i)

        Returns
        -------
        w : (L, F) complex array
        """
        return np.exp(-1j * np.outer(delays, self.omega))  # (L, F)

    def beamform(self, signal_matrix, target_point):
        """
        Apply delay-and-sum beamformer to a signal matrix of shape (T, F, L)

        Parameters
        ----------
        signal_matrix : array
            Raw IQ data of shape (T, F, L)
            - T: slow-time samples (pulses)
            - F: fast-time samples (frequency bins)
            - L: number of channels/elements
        target_point : (3,) array
            3D focus point

        Returns
        -------
        B : (T, F) complex array
            Beamformed signal (collapsed across channels), same slow-fast ordering
        """
        # Compute delays and weights
        delays = self._compute_delays(target_point)        # (L,)
        w = self._compute_weights(delays)                  # (L, F)
        # Transpose to shape (F, L) for broadcasting
        w_fl = w.T                                        # (F, L)
        # Multiply weights and sum across channels
        # signal_matrix: (T, F, L)
        # w_fl:            (F, L)
        # broadcast to (T, F, L)
        weighted = signal_matrix * w_fl[None, :, :]       # (T, F, L)
        # Sum over channel axis
        return np.sum(weighted, axis=2)                   # (T, F)
