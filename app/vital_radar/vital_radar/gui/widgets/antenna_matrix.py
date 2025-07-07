from PyQt6.QtWidgets import (
    QWidget, QGridLayout, QLabel, QCheckBox, QSizePolicy, QLayout
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer


# dictionary to map which antenna pairs exist, in the desired order
tx_to_rx = {
    1: [2, 3, 6, 7, 10, 11, 14, 15, 4, 8, 12, 16, 18],
    4: [2, 3, 6, 7, 10, 11, 14, 15],
    17: [2, 3, 6, 7, 10, 11, 14, 15, 8, 12, 16],
    18: [2, 3, 6, 7, 10, 11, 14, 15],
}

class AntennaMatrix(QWidget):
    """
    A checkbox matrix for selecting antenna pairs.
    Emits selectionChanged(tx, rx, checked) on every toggle.
    """
    selectionChanged = pyqtSignal(int, int, bool)

    def __init__(self, tx_to_rx: dict[int, list[int]], parent=None):
        super().__init__(parent)

        # 
        self.tx_list = list(tx_to_rx.keys())
        
        seen = {}
        for rxs in tx_to_rx.values():
            for rx in rxs:
                seen.setdefault(rx, True)
        self.rx_list = list(seen.keys())

        self._checkboxes: dict[tuple[int,int], QCheckBox] = {}

        grid = QGridLayout()
        grid.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)
        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        grid.setAlignment(Qt.AlignmentFlag.AlignTop)

        # 'RX' head spanning all RX columns
        rx_head = QLabel("RX")
        grid.addWidget(rx_head, 0, 1)

        # 'TX' head spanning all TX rows
        tx_head = QLabel("TX")
        grid.addWidget(tx_head, 1, 0)

        # numeric RX column headers
        for col, rx in enumerate(self.rx_list, start=2):
            lbl = QLabel(str(rx))
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(lbl, 0, col)

        # numeric TX row headers
        for row, tx in enumerate(self.tx_list, start=2):
            lbl = QLabel(str(tx))
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(lbl, row, 0)

        # checkboxes in the body
        for r, tx in enumerate(self.tx_list, start=2):
            allowed = set(tx_to_rx.get(tx, []))
            for c, rx in enumerate(self.rx_list, start=2):
                if rx in allowed:
                    cb = QCheckBox()
                    cb.toggled.connect(
                        lambda checked, tx=tx, rx=rx:
                            self.selectionChanged.emit(tx, rx, checked)
                    )
                    grid.addWidget(cb, r, c, alignment=Qt.AlignmentFlag.AlignCenter)
                    self._checkboxes[(tx, rx)] = cb
                else:
                    grid.addWidget(QLabel(""), r, c)

        self.setLayout(grid)

    def is_checked(self, tx: int, rx: int) -> bool:
        return self._checkboxes.get((tx, rx), QCheckBox()).isChecked()

    def apply_defaults(self, defaults: list[tuple[int,int]]):
        QTimer.singleShot(0, lambda: self._click_defaults(defaults))

    def _click_defaults(self, defaults):
        """
        Simulate a user-click on each default pair.

        """
        for tx, rx in defaults:
            cb = self._checkboxes.get((tx, rx))
            if cb and not cb.isChecked():
                cb.click()
                