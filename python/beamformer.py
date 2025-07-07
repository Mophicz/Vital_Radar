import numpy as np
    
    
from scipy.constants import c


class DelaySumBeamformer:
    """
    A simple delay-and-sum beamformer assuming signal_matrix is (slow-time, fast-time, channels).

    """
    def __init__(self, element_positions, frequencies):
        self.positions = np.asarray(element_positions)  
        self.freqs = np.asarray(frequencies)           
        
        # Precompute angular frequencies
        self.omega = 2 * np.pi * self.freqs           

    def _compute_delays(self, target_point):
        """
        Compute one-way delays from each element to the target and back
        
        """
        # distances: shape (L,)
        d = np.linalg.norm(self.positions - target_point[None, :], axis=1)
        return 2 * d / c  

    def _compute_weights(self, delays):
        """
        Steering weights per element and frequency

        """
        return np.exp(-1j * np.outer(delays, self.omega)) 

    def beamform(self, signal_matrix, target_point):
        """
        Apply delay-and-sum beamformer to a signal matrix

        """
        # Compute delays and weights
        delays = self._compute_delays(target_point)      
        w = self._compute_weights(delays)    
                     
        # Transpose
        w_fl = w.T     
                                       
        # Multiply weights and sum across channels
        weighted = signal_matrix * w_fl[None, :, :]      
        
        # Sum over channel axis
        return np.sum(weighted, axis=2)                 
  