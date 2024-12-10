import pandas as pd
import numpy as np
import os
import shutil
from sklearn.model_selection import train_test_split
from globals import *


# Prepare the large file and empty the small file
data_path = "732A76_Research_Project/Data"

try:
    os.remove(data_path + "small.csv")
    os.remove(data_path + "large.csv")
    shutil.copy(data_path + "medium.csv", data_path + "large.csv")
except OSError:
    pass

# Load the data
chunksize = 100000
df_chunks = pd.read_csv(data_path + "medium.csv", names=col_names, chunksize=chunksize)

# Check the number of rows
n_rows = 11000000 # sum(len(chunk.index) for chunk in df_chunks)
print(f"{n_rows} rows")

# Reset the iterator
# df_chunks = pd.read_csv(data_path + "medium.csv", names=col_names, chunksize=chunksize)

train_ind = np.random.choice(n_rows, size=int(n_rows * 0.8))

# Iterate over chunks
for i, chunk in enumerate(df_chunks):
    # Half the dataset into small.csv
    if chunksize * (i + 1) <= n_rows / 2:
        # In this case, only the size is important, but otherwise should sample randomly
        chunk.to_csv(data_path + "small.csv", mode="a")

    chunk.to_csv(data_path + "large.csv", mode="a")
