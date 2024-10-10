#%%
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from tqdm import tqdm

data_path = r"C:\Users\marij\Documents\Courses\Fall 2024\732A76 - Research Project\HIGGS.csv"
names = ["label", "lepton_pT", "lepton_eta", "lepton_phi", 
         "missing_energy_mag", "missing_energy_phi", 
         "jet_1_pt", "jet_1_eta", "jet_1_phi", "jet_1_b_tag",
         "jet_2_pt", "jet_2_eta", "jet_2_phi", "jet_2_b_tag",
         "jet_3_pt", "jet_3_eta", "jet_3_phi", "jet_3_b_tag",
         "jet_4_pt", "jet_4_eta", "jet_4_phi", "jet_4_b_tag",
         "m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "b_wbb", "m_wwbb",
         ]

# Number of rows
# print(f"No. rows: {sum(1 for row in open(data_path, 'r'))}")

df_chunks = pd.read_csv(data_path, names=names, chunksize=100000)

coef = np.zeros((len(names),))
for i, df in tqdm(enumerate(df_chunks)):
    X = df.drop("label", axis=1)
    y = df.label

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)

    lin_reg = LinearRegression(n_jobs=-1)
    lin_reg.fit(X_train, y_train)

    coef += np.insert(lin_reg.coef_, 0, lin_reg.intercept_)


coef /= (i + 1)

y_train_pred = np.c_[np.ones(len(y_train)), X_train] @ coef
y_test_pred = np.c_[np.ones(len(y_test)), X_test] @ coef
train_err = root_mean_squared_error(y_train, y_train_pred)
test_err = root_mean_squared_error(y_test, y_test_pred)

# %%
