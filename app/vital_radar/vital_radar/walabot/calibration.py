from PyQt6.QtCore import QThread
import WalabotAPI as wlbt


class CalibrationWorker(QThread):
    def run(self):
        wlbt.StartCalibration()
        stat, prog = wlbt.GetStatus()
        while stat == wlbt.STATUS_CALIBRATING and prog < 100:
            wlbt.Trigger()
            stat, prog = wlbt.GetStatus()
            print(f"Calibrating {prog}%")
            self.msleep(100)
        print("Calibration complete")
        