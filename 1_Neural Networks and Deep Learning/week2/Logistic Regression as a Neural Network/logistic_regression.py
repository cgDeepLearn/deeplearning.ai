# -*- coding: utf-8 -*-
"""
Created on Sun Sep 08 15:39:30 2017
Following the course of Andrew.Ng, DeepLearning I
First step a single neuron -- logistic regression
@author: cgDeepLearn
"""
import numpy as np
import pandas as pd


def sigmoid(Z):
    """sigmoid activate function
    # A problem of log ..np.exp----->0,bigfloat
    """
    A = 1.0 / (1 + np.exp(-Z))
    return A


def train(epoch_limits, alpha, W, train_set, y):
    """
    logistic train
    """
    J = []  # pylint: disable=C0103
    assert W.shape[1] == train_set.shape[0]
    for step in range(epoch_limits):
        Z = np.dot(W, train_set)  # pylint: disable=C0103
        A = sigmoid(Z)  # pylint: disable=C0103

        a_first = np.log(A)
        a_second = np.log(1 - A)
        L_first = np.dot(a_first, y)
        L_second = np.dot(a_second, 1 - y)
        L = (L_first + L_second) * -1
        J.append(L[0][0])

        dZ = A - y.T
        dW = np.dot(dZ, train_set.T) * 1.0 / train_set.shape[1]
        W = W - alpha * (1.0 - (1 + step) / epoch_limits) * dW
        print("epoch: %05d cost %.02f" % (step, L[0][0]))
        precision = test(train_set, y, W)
        print("precision on train set: %.02f" % precision)
    return W


def predict(X, W):
    assert X.shape[0] == W.shape[1] 
    y = sigmoid(np.dot(W, X))
    y[y > 0.5] = 1
    y[y <= 0.5] = 0
    return y


def loadData():
    train_set = pd.read_csv("X.csv", header=None)
    train_label = pd.read_csv("y.csv", header=None)
    test_set = pd.read_csv("Tx.csv", header=None)
    test_label = pd.read_csv("Ty.csv", header=None)
    return train_set, train_label, test_set, test_label


def test(val_set, val_label, params):
    result = predict(val_set, params)
    return sum(result.T == val_label) * 1.0 / val_set.shape[1]


def do():
    train_set, train_label, test_set, test_label = loadData()
    # add a bias columns
    train_set.loc[:, 'bias'] = pd.Series(np.ones(train_set.shape[0]))
    test_set.loc[:, 'bias'] = pd.Series(np.ones(test_set.shape[0]))
    # initial params
    W = np.random.randn(1, train_set.shape[1])
    alpha = 0.01
    epoch = 40000
    W = train(epoch, alpha, W, train_set.values.T, train_label.values)
    print("Test precision: %.2f" %
          test(test_set.values.T, test_label.values, W))

if __name__ == '__main__':
    do()
