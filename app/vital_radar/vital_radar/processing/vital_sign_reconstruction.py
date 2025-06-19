import numpy as np
from scipy.constants import c

B = 1.7e9      # bandwidth
K = 137
F0 = 6.3e9

def steeringVector(d):
    dF = B/K
    k = np.arange(K)

    a = np.exp(-1j * 2 * np.pi * (F0 + k * dF) * 2 * d / c)
    
    return a


def computeWeights(a):
    w = a @ np.linalg.inv(a.conj().T @ a)
    
    return w


def spatialFilter(X, d):
    
    sensor_positions = [
        [-10,56,0],
        [-10,36,0],
        [-10,16,0],
        [-10,-4,0],
        [-10,136,0], 
        [-10,116,0], 
        [-10,96,0], 
        [-10,76,0]
    ]
    sensor_positions = np.array(sensor_positions) * 1e-6
    
    target_position = [
        0,0,d
    ]
    
    target_position = np.array(target_position)
    
    a = steeringVector(d)
    
    w = computeWeights(a)
    
    return w