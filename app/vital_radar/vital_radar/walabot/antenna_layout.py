import numpy as np


class AntennaLayout:
    """
    Represents the 3D coordinates of Tx and Rx antennas.

    Attributes
    ----------
    tx_positions : dict[int, np.ndarray]
        Maps Tx antenna index (1-based) to its (x, y, z) position in meters.
    rx_positions : dict[int, np.ndarray]
        Maps Rx antenna index (1-based) to its (x, y, z) position in meters.
    """
    def __init__(self, tx_positions, rx_positions):
        # Expect dicts mapping 1-based antenna IDs to 3-element tuples or arrays
        self.tx_positions = {i: np.asarray(pos) for i, pos in tx_positions.items()}
        self.rx_positions = {i: np.asarray(pos) for i, pos in rx_positions.items()}

    def get_channel_positions(self, pair_list):
        """
        Return the virtual element positions for the specified tx/rx pairs.

        For each (tx, rx) tuple in pair_list, the virtual channel position is defined as:
            (tx_pos + rx_pos) / 2
        or you can choose to treat them separately for bistatic beamforming.

        Parameters
        ----------
        pair_list : list[tuple[int, int]]
            List of (tx, rx) antenna ID pairs to include.

        Returns
        -------
        positions : np.ndarray
            Array of shape (M, 3) where M = len(pair_list),
            giving the virtual-element positions for each pair in row-major order.
        pairs : list[tuple[int, int]]
            Echo of the input list, corresponding to each row in positions.
        """
        if pair_list:
            positions = []

            for tx, rx in pair_list:
                tx_pos = self.tx_positions.get(tx)
                if tx_pos is None:
                    raise KeyError(f"Tx antenna {tx} not found in layout")

                rx_pos = self.rx_positions.get(rx)
                if rx_pos is None:
                    raise KeyError(f"Rx antenna {rx} not found in layout")

                # Virtual channel at midpoint
                virt_pos = (tx_pos + rx_pos) / 2
                positions.append(virt_pos)

            positions = np.vstack(positions)
            return positions, list(pair_list)
        else:
            return None


# Example: define your antenna grid (in meters)
# Here you must fill in the actual coordinates for each numbered antenna
# following your system's datasheet or the diagram.
# Coordinates are (x, y, z) in meters.
POS_TX = {
    1: (-0.03,  0.056, 0.0),
    4: (0.03,  0.056, 0.0),
    17: (-0.03, -0.024, 0.0),
    18: (0.03, -0.024, 0.0)
}
POS_RX = {
    2: (-0.01,  0.056, 0.0),
    3: (0.01, 0.056, 0.0),
    6: (-0.01, 0.036, 0.0),
    7: (0.01, 0.036, 0.0),
    10: (-0.01, 0.016, 0.0),
    11: (0.01, 0.016, 0.0),
    14: (-0.01, -0.004, 0.0),
    15: (0.01, -0.004, 0.0),
    4: (0.03,  0.056, 0.0),
    8: (0.03,  0.036, 0.0),
    12: (0.03,  0.016, 0.0),
    16: (0.03,  -0.004, 0.0),
    18: (0.03,  -0.024, 0.0)
}

# Instantiate a default layout:
# default_layout = AntennaLayout(ANT_TX, ANT_RX)

# Usage in beamforming:
# tx_ids = [1,3,5]
# rx_ids = [2,4]
# positions, channel_pairs = default_layout.get_channel_positions(tx_ids, rx_ids)
# # positions is now an array of shape (3*2, 3) for your beamformer

antenna_layout = AntennaLayout(POS_TX, POS_RX)
