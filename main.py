# want to use a dataset for Machine Learning

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

index = range(32)
dataset = pd.read_csv("wdbc.csv", names=index)
print(dataset.head())

dataset = dataset.drop(0, axis=1)

print(dataset.head())

X = dataset.drop(1, axis=1)
y = dataset[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

RBF = RandomForestClassifier(n_estimators=100)
score = cross_val_score(RBF, X, y)
print('random forest score: %.5f' % score.mean())

svlin = SVC(kernel='linear', C=10.0)
score = cross_val_score(svlin, X, y)
print('svc linear score: %.5f' % score.mean())

svlin = SVC(kernel='poly', C=10.0, degree=3)
score = cross_val_score(svlin, X, y)
print('svc poly score: %.5f' % score.mean())

svlin = SVC(kernel='rbf', C=10.0, gamma='auto')
score = cross_val_score(svlin, X, y)
print('svc rbf score: %.5f' % score.mean())

# implement a multi layer perceptron
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(1000, 100, 10), max_iter=1000)
score = cross_val_score(mlp, X, y)
print('mlp score: %.5f' % score.mean())
