#%% Process data
import dask.dataframe as dd
from dask.distributed import Client
from dask_ml.model_selection import train_test_split
from dask.diagnostics import ResourceProfiler
import time
from globals import *

client = Client()
print(client)

with ResourceProfiler() as rprof:
    for data_path in [small_path]:
        # Data not actually loaded yet, just partitioned and read labels
        # https://tutorial.dask.org/01_dataframe.html
        # Setting lengths=True computes chunk sizes, needed with arrays to allow slicing but takes some time
        # https://docs.dask.org/en/stable/array-creation.html
        # start_time_data = time.perf_counter()

        # with Profiler() as prof, ResourceProfiler() as rprof:
        darr = dd.read_csv(data_path, header=None).repartition().to_dask_array(lengths=True)
        darr.visualize(filename="graph.png")

        # Split data into train/test
        X = darr[:, 1:] # Everything except columns with class labels
        y = darr[:, 0] # Class labels
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

        # end_time_data = time.perf_counter()
        # print(f"Data prep time: {end_time_data - start_time_data:.3f}")

#%% Linear regression
from dask_ml.linear_model import LinearRegression

start_time_lin_reg = time.perf_counter()

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)
train_score = lin_reg.score(X_train, y_train)
test_score = lin_reg.score(X_test, y_test)

print(f"Train score: {train_score:.3f}\nTest score: {test_score:.3f}")

end_time_lin_reg = time.perf_counter()
print(f"Linear regression time: {end_time_lin_reg - start_time_lin_reg:.3f}")

#%% Logistic regression
from dask_ml.linear_model import LogisticRegression

start_time_log_reg = time.perf_counter()

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
train_score = log_reg.score(X_train, y_train)
test_score = log_reg.score(X_test, y_test)

print(f"Train score: {train_score:.3f}\nTest score: {test_score:.3f}")

end_time_log_reg = time.perf_counter()
print(f"Logistic regression time: {end_time_log_reg - start_time_log_reg:.3f}")

#%% Decision tree
from dask_ml.xgboost import XGBClassifier

start_time_dec_tree = time.perf_counter()

dec_tree = XGBClassifier()
dec_tree.fit(X_train, y_train)
train_score = dec_tree.score(X_train, y_train)
test_score = dec_tree.score(X_test, y_test)

print(f"Train score: {train_score:.3f}\nTest score: {test_score:.3f}")

end_time_dec_tree = time.perf_counter()
print(f"Decision tree time: {end_time_dec_tree - start_time_dec_tree:.3f}")

#%% Clustering
from dask_ml.cluster import KMeans
from dask_ml.metrics import accuracy_score

start_time_kmeans = time.perf_counter()

kmeans = KMeans(n_clusters=2)
kmeans.fit(X_train)
cluster_pred = kmeans.predict(X_test)
acc = accuracy_score(y_test, cluster_pred)

print(f"Test accuracy: {acc:.3f}")

end_time_kmeans = time.perf_counter()
print(f"Decision tree time: {end_time_kmeans - start_time_kmeans:.3f}")
