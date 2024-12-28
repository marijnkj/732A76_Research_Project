import pandas as pd
import numpy as np
import os
import shutil
from globals import *
from tqdm import tqdm

# Prepare the large file and empty the small file
data_path = "C:/Users/marij/Documents/GitHub/732A76_Research_Project/Data/"

try:
    os.remove(data_path + "train_small.csv")
except OSError:
    pass
try:
    os.remove(data_path + "test_small.csv")
except OSError:
    pass
try:
    os.remove(data_path + "train_medium.csv")
except OSError:
    pass
try:
    os.remove(data_path + "test_medium.csv")
except OSError:
    pass
try:
    os.remove(data_path + "train_large.csv")
except OSError:
    pass
try:
    os.remove(data_path + "test_large.csv")
except OSError:
    pass

shutil.copy(data_path + "HIGGS.csv", data_path + "train_medium.csv")
shutil.copy(data_path + "HIGGS.csv", data_path + "train_large.csv")
print("Init done")

# Load the data
chunksize = 300000
df_chunks = pd.read_csv(data_path + "HIGGS.csv", names=col_names, chunksize=chunksize)

# Check the number of rows
n_rows = 11000000 # sum(len(chunk.index) for chunk in df_chunks)
print(f"{n_rows} rows")

# Reset the iterator
# df_chunks = pd.read_csv(data_path + "HIGGS.csv", names=col_names, chunksize=chunksize)

# Sample test sample index
test_ind = np.random.choice(n_rows, size=int(n_rows * 0.2))

# Iterate over chunks
for i, chunk in tqdm(enumerate(df_chunks)):
    chunk = chunk.astype({col: float for col in col_names})
    this_chunk_test_ind = [ind for ind in test_ind if ind in chunk.index]
    
    # Half the dataset into small.csv
    if chunksize * (i + 1) <= n_rows / 2:
        chunk.to_csv(data_path + "train_small.csv", mode="a")
        chunk.loc[this_chunk_test_ind, :].to_csv(data_path + "test_small.csv", mode="a")

    # Append into medium file
    chunk.loc[this_chunk_test_ind, :].to_csv(data_path + "test_medium.csv", mode="a")

    # Append twice into large file
    chunk.to_csv(data_path + "train_large.csv", mode="a")
    pd.concat([chunk.loc[this_chunk_test_ind, :]] * 2).to_csv(data_path + "test_large.csv", mode="a")
    
    