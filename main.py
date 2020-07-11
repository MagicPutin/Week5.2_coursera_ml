import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm._libsvm import predict_proba

# data preparing

data = pd.read_csv('Data/gbm-data.csv')
X = data.loc[:, data.columns != 'Activity']
y = data['Activity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)

# classification
for i in [1, 0.5, 0.3, 0.2, 0.1]:
    clf = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate=i)
    clf.fit(X_train, y_train)
    test_loss = np.empty(250)
    for m, y_decision in enumerate(clf.staged_decision_function(X_test)):
        y_pred_test = 1.0 / (1.0 + np.exp(-y_decision))
        test_loss[m] = log_loss(y_test, y_pred_test)

    train_loss = np.empty(250)
    for m, y_decision in enumerate(clf.staged_decision_function(X_train)):
        y_pred_train = 1.0 / (1.0 + np.exp(-y_decision))
        train_loss[m] = log_loss(y_train, y_pred_train)
    if i == 0.2:
        iter = min(range(len(test_loss)), key=test_loss.__getitem__)
        val = min(test_loss)
    plt.figure()
    plt.plot(test_loss, 'r', linewidth=2)
    plt.plot(train_loss, 'g', linewidth=2)
    plt.legend(['test', 'train'])
    plt.show()

with open('Answers/task2.txt', 'w') as task2:
    task2.write(str(val) + ' ' + str(iter))

# 3rd task
clf = RandomForestClassifier(n_estimators=iter, random_state=241)
clf.fit(X_train, y_train)
prediction = clf.predict_proba(X_test)
loss = log_loss(y_test, prediction)
with open('Answers/task3.txt') as task3:
    task3.write(str(round(loss, 2)))
