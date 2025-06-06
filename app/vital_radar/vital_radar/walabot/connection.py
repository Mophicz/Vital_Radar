import WalabotAPI as wlbt

def init_radar():
    """
    Initialize and start Walabot. 
    Raises an exception if anything fails.
    """
    wlbt.Init()
    wlbt.Initialize()
    wlbt.ConnectAny()
    wlbt.SetProfile(wlbt.PROF_SENSOR)
    wlbt.SetDynamicImageFilter(wlbt.FILTER_TYPE_NONE)
    wlbt.Start()
    return True 

def stop_radar():
    """
    Stop and disconnect from Walabot. If something
    is already stopped, it will just pass.
    """
    try:
        wlbt.Stop()
        wlbt.Disconnect()
    except Exception:
        pass
    finally:
        wlbt.Clean()
    return True

def reconnect_radar():
    """
    A convenience function that does stop_radar()
    then init_radar(). You can also catch exceptions
    and return a boolean.
    """
    stop_radar()
    return init_radar()
