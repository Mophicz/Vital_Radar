import time

import numpy as np
import WalabotAPI as wlbt


trigger_freq = float('nan')


def updateTriggerFreq(alpha=0.3, _state={'last': None, 'ema_dt': None}):
    """
    Memory for 1) latest trigger time and 2) Exponential Moving Average (EMA) of time difference between triggers. 
    Updates module level state 'trigger_freq' with the EMA trigger‐frequency in Hz if called once per 'wlbt.Trigger()'.
    """
    global trigger_freq
    
    # get current time
    now = time.perf_counter()
    
    # temp variable for last time
    last = _state['last']
    
    if last is not None:
        # time difference between 2 Triggers
        dt = now - last
        
        # temp variable for current average time difference
        ema_dt = _state['ema_dt']
        
        # add current time to moving average
        _state['ema_dt'] = dt if ema_dt is None else alpha*dt + (1-alpha)*ema_dt
    
    # current time is now last time
    _state['last'] = now

    # compute frequency as reciprocal of time difference
    avg = _state['ema_dt']
    
    # update trigger frequency
    trigger_freq = (1.0/avg) if (avg is not None and avg > 0) else float('nan')


def getSignals(pairs_list):
    """
    Triggers the radar and retrieves signal matrix for given TX/RX antenna combinations.

    Returns:
        signals: 2D numpy array (fast-time x channels) or None if radar error.
    """
    try:
        wlbt.Trigger()
        
        # update trigger fequency
        updateTriggerFreq()
        
        pairs = wlbt.GetAntennaPairs()
        signals = None

        for tx, rx in pairs_list:
            pair = next(
                (p for p in pairs
                 if p.txAntenna == tx and p.rxAntenna == rx),
                None
            )
            if pair is None:
                continue

            sig, _ = wlbt.GetSignal(pair)
            sig = np.array(sig)             # shape: (fast_time,)
            col = sig[:, np.newaxis]        # shape: (fast_time, 1)
            signals = col if signals is None else np.concatenate((signals, col), axis=1)

        return signals
    
    except Exception as e:
        print("Failed to get signal matrix:", e)
        return None
    