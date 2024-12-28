from globals import *
import vaex
from vaex.ml.sklearn import IncrementalPredictor, Predictor
from sklearn.linear_model import LinearRegression

train_small_path = "Data/train_small.csv"
test_small_path = "732A76_Research_Project/Data/test_small.csv"
train_medium_path = "732A76_Research_Project/Data/train_medium.csv"
test_medium_path = "732A76_Research_Project/Data/test_medium.csv"
train_large_path = "732A76_Research_Project/Data/train_large.csv"
test_large_path = "732A76_Research_Project/Data/test_large.csv"
col_names = ["label", "lepton_pT", "lepton_eta", "lepton_phi", 
             "missing_energy_mag", "missing_energy_phi", 
             "jet_1_pt", "jet_1_eta", "jet_1_phi", "jet_1_b_tag",
             "jet_2_pt", "jet_2_eta", "jet_2_phi", "jet_2_b_tag",
             "jet_3_pt", "jet_3_eta", "jet_3_phi", "jet_3_b_tag",
             "jet_4_pt", "jet_4_eta", "jet_4_phi", "jet_4_b_tag",
             "m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "b_wbb", "m_wwbb",
             ]

features = [col_name for col_name in col_names if col_name != "label"]

df = vaex.open(train_small_path, dtype={col_name: "float" for col_name in col_names}) # convert=train_small_path.replace(".csv", ".parquet")
for column in df.columns:
    if df[column].dtype == "object":
        df[column] = df[column].astype("float")

df = vaex.open(fr"C:\Users\marij\Documents\GitHub\732A76_Research_Project\Data\small.parquet")
model = LinearRegression()
vaex_model = Predictor(features=features, target="label", model=model, delay=True)
vaex_model.fit(df, progress="widget", prediction_name="prediction", delay=True)



# vaex_model = IncrementalPredictor(features=features, target="label", model=model, batch_size=10_000)

print("done")

import numpy as np