import pandas as pd
import numpy as np

# 1) grab the sampling rate from the first line
csv_path = "signal_buffer_with_fs.csv"
with open(csv_path, "r") as f:
    header = f.readline().strip()
# header looks like "# fs = 123.456"
fs = float(header.lstrip("# ").split("=")[1])

# 2) read the DataFrame, skipping comment lines and using a two‐row header
df = pd.read_csv(
    csv_path,
    comment="#",          # ignore lines starting with #
    header=[0,1],         # two header rows -> MultiIndex
    index_col=0           # first column was the sample index
)
# now df.shape == (M, 137*4)

# 3) reshape back into (time, channels, pairs)
M, flat_cols = df.shape
n_channels = 137
n_pairs    = flat_cols // n_channels

arr3d = df.values.reshape(M, n_channels, n_pairs)

# sanity check
assert n_pairs == 4
print("Recovered array shape:", arr3d.shape)
print("Recovered fs:", fs)
