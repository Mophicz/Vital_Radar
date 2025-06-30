from PyQt6.QtWidgets import (
    QWidget, QGridLayout, QLabel, QCheckBox, QSizePolicy, QLayout
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer


tx_to_rx = {
    1: [2, 3, 6, 7, 10, 11, 14, 15, 4, 8, 12, 16, 18],
    4: [2, 3, 6, 7, 10, 11, 14, 15],
    17: [2, 3, 6, 7, 10, 11, 14, 15, 8, 12, 16],
    18: [2, 3, 6, 7, 10, 11, 14, 15],
}

class AntennaMatrix(QWidget):
    """
    A TX×RX checkbox matrix with a single 'TX' and 'RX' header.
    Emits selectionChanged(tx, rx, checked) on every toggle.
    """
    selectionChanged = pyqtSignal(int, int, bool)

    def __init__(self, tx_to_rx: dict[int, list[int]], parent=None):
        super().__init__(parent)

        # preserve insertion order for TX and RX
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

        # 1) Single 'RX' head spanning all RX columns
        rx_head = QLabel("RX")
        grid.addWidget(rx_head, 0, 1)

        # 2) Single 'TX' head spanning all TX rows
        tx_head = QLabel("TX")
        grid.addWidget(tx_head, 1, 0)

        # 3) Numeric RX column headers
        for col, rx in enumerate(self.rx_list, start=2):
            lbl = QLabel(str(rx))
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(lbl, 0, col)

        # 4) Numeric TX row headers (col 0, rows 2…)
        for row, tx in enumerate(self.tx_list, start=2):
            lbl = QLabel(str(tx))
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            grid.addWidget(lbl, row, 0)

        # 5) Checkboxes in the body (rows 2…, cols 1…)
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
        """
        Simulate a user-click on each default pair so that
        - the checkbox is checked
        - both toggled() and clicked() signals fire
        - your selectionChanged → onMatrixChange logic runs exactly as for a real click
        """
        # We use a singleShot(0) so this happens *after* the widget is fully set up
        QTimer.singleShot(0, lambda: self._click_defaults(defaults))

    def _click_defaults(self, defaults):
        for tx, rx in defaults:
            cb = self._checkboxes.get((tx, rx))
            if cb and not cb.isChecked():
                cb.click()    # ← this is the magic: emits clicked() then toggled()
                