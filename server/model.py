import pandas as pnd
import numpy as np

from dask.distributed import Client, progress
client = Client(processes=False, threads_per_worker=4,
                n_workers=1, memory_limit='2GB')

import sys
from  pathlib import Path

from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics

sys.path.append(str(Path.cwd))

data = pnd.read_csv('./data/books_data.csv', sep=',', encoding='latin-1', header=0, on_bad_lines='skip')
data = data.iloc[:, 5:].to_numpy()
X = data[:, :-1]
Y = data[:, -1]

scaler = preprocessing.MinMaxScaler().fit(X)

#X = preprocessing.normalize(X, axis=0, norm="max")
X = scaler.transform(X)

def get_model():
    import joblib

    model = None

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    with joblib.parallel_backend('dask'):
        model = Perceptron(penalty="l1", alpha=0.0002, warm_start=True)
        model.fit(X_train, Y_train)
    return model

def scale(data):
  print(X[0])
  print(f"DATA in SCALE = {data}")
  return scaler.transform(data)

def get_stats(model):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    predictions = model.predict(X_test)
    acc = metrics.accuracy_score(Y_test, predictions)
    mae = metrics.mean_absolute_error(Y_test, predictions)

    mse = metrics.mean_squared_error(Y_test, predictions)
    rmse = np.sqrt(mse)

    fpr, tpr, thresholds = metrics.roc_curve(Y_test, predictions)
    auc = metrics.auc(fpr, tpr)

    loss = metrics.log_loss(Y_test, predictions, eps=1e-15)

    return {
        "ACC": acc*100,
        "LOSS": loss,
        "RMSE": rmse,
        "AUC": auc * 100,
        "MAE": mae, 
    }

# print(f"ACC:{acc*100:.2f} \nLOSS:{loss:.2f} \nRMSE:{rmse:.2f} \nAUC:{auc*100:.2f} \nMAE:{mae:.2f}")
#loss = metrics.pairwise_distances(X, Y, metric='euclidean')

