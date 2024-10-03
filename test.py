import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data_path = r"C:\Users\marij\Documents\Courses\Fall 2024\732A76 - Research Project\HIGGS.csv"
names = ["label", "lepton_pT", "lepton_eta", "lepton_phi", 
         "missing_energy_mag", "missing_energy_phi", 
         "jet_1_pt", "jet_1_eta", "jet_1_phi", "jet_1_b_tag",
         "jet_2_pt", "jet_2_eta", "jet_2_phi", "jet_2_b_tag",
         "jet_3_pt", "jet_3_eta", "jet_3_phi", "jet_3_b_tag",
         "jet_4_pt", "jet_4_eta", "jet_4_phi", "jet_4_b_tag",
         "m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "b_wbb", "m_wwbb",
         ]

start_time1 = time.perf_counter()

df_full = pd.read_csv(data_path, names=names)
X_train, X_test, y_train, y_test = train_test_split(df_full.drop("label", axis=1), df_full.label, train_size=0.8)
linreg = LinearRegression().fit(df_full.drop("label", axis=1), df_full.label)
y_pred = linreg.predict(X_test)
mean_squared_error(y_test, y_pred)

end_time1 = time.perf_counter()

print(f"Standard Pandas time: {end_time1 - start_time1:.3f}")

# start_time2 = time.perf_counter()
# df_chunked = pd.read_csv(data_path, chunksize=50)
# end_time2 = time.perf_counter()

# print(f"Time to read full df: {end_time1 - start_time1:.3f}\nTime to read chunked df: {end_time2 - start_time2:.3f}")

# df_chunked

import dask.dataframe as dd
import dask.array as da
from dask.distributed import Client
from dask_ml.linear_model import LinearRegression, LogisticRegression
from dask_ml.model_selection import train_test_split
from dask_ml.metrics import mean_squared_error

client = Client()
client

# Data not actually loaded yet, just partitioned and read labels
# https://tutorial.dask.org/01_dataframe.html
# Setting lengths=True computes chunk sizes, needed with arrays to allow slicing but takes some time
# https://docs.dask.org/en/stable/array-creation.html
start_time2 = time.perf_counter()
darr = dd.read_csv(data_path, header=None).repartition(npartitions=12).to_dask_array(lengths=True)
# darr.visualize(filename="graph.png")

X = darr[:, 1:] # Everything except columns with class labels
y = darr[:, 0] # Class labels
X_train, X_test, y_train, y_test = train_test_split(darr[:, 1:], darr[:, 0], train_size=0.8)

linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred = linreg.predict(X_test)
mean_squared_error(y_test, y_pred)

end_time2 = time.perf_counter()

print(f"Dask time: {end_time2 - start_time2:.3f}")

# logrg = LogisticRegression()
# linreg.fit(X_train.to_dask_array(), y_train.to_dask_array())