#%%
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from tqdm import tqdm
from globals import *

df_chunks = pd.read_csv(small_path, names=col_names, chunksize=100000)

coef = np.zeros((len(col_names),))
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
