import numpy as np


class AntennaLayout:
    """
    Represents the 3D coordinates of Tx and Rx antennas.

    """
    def __init__(self, tx_positions, rx_positions):
        # Expect dicts mapping 1-based antenna IDs to 3-element tuples or arrays
        self.tx_positions = {i: np.asarray(pos) for i, pos in tx_positions.items()}
        self.rx_positions = {i: np.asarray(pos) for i, pos in rx_positions.items()}

    def get_channel_positions(self, pair_list):
        """
        Return the virtual element positions for the specified tx/rx pairs.
        
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


# define antenna grid 
# coordinates are (x, y, z) in meters
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

antenna_layout = AntennaLayout(POS_TX, POS_RX)
