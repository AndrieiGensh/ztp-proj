import pandas as pnd
import numpy as np

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

X = preprocessing.normalize(X, axis=0, norm="max")

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=111)

model = Perceptron(penalty="l1", alpha=0.0002, warm_start=True)

model.fit(X_train, Y_train)

predictions = model.predict(X_test)
acc = metrics.accuracy_score(Y_test, predictions)
mae = metrics.mean_absolute_error(Y_test, predictions)

mse = metrics.mean_squared_error(Y_test, predictions)
rmse = np.sqrt(mse)

fpr, tpr, thresholds = metrics.roc_curve(Y_test, predictions)
auc = metrics.auc(fpr, tpr)


loss = metrics.log_loss(Y_test, predictions, eps=1e-15)

print(f"ACC:{acc*100:.2f} \nLOSS:{loss:.2f} \nRMSE:{rmse:.2f} \nAUC:{auc*100:.2f} \nMAE:{mae:.2f}")
#loss = metrics.pairwise_distances(X, Y, metric='euclidean')


import pickle

pickle.dump(model, open("./model/modeldata.sav", 'wb'))

