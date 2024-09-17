#%% Loading libraries
import pandas as pd
import numpy as np
import time

#%% 
data_path = r"C:\Users\marij\Documents\Courses\Fall 2024\732A76 - Research Project\HIGGS.csv"

start_time1 = time.perf_counter()
df_full = pd.read_csv(data_path)
end_time1 = time.perf_counter()

start_time2 = time.perf_counter()
df_chunked = pd.read_csv(data_path, chunksize=50)
end_time2 = time.perf_counter()

print(f"Time to read full df: {end_time1 - start_time1:.3f}\nTime to read chunked df: {end_time2 - start_time2:.3f}")
#%%
df_chunked

# %%
