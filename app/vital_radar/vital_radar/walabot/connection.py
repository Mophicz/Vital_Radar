import WalabotAPI as wlbt


def initRadar():
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


def stopRadar():
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


def reconnectRadar():
    """
    A convenience function that does stop_radar()
    then init_radar(). You can also catch exceptions
    and return a boolean.
    """
    stopRadar()
    return initRadar()
