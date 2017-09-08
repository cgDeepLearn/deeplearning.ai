# -*- coding: utf-8 -*-
"""
Created on Sun Sep 08 15:39:30 2017
Following the course of Andrew.Ng, DeepLearning I
First step a single neuron -- logistic regression
@author: cgDeepLearn
"""


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def sigmoid(Z):
    """
    Compute the sigmoid of z
    Arguments:
    Z -- A scalar or numpy array of any size.
    Return:
    s -- sigmiod(Z)
    """
    s = 1. / (1. + np.exp(-Z))
    return s


class LRmodel():
    """
    """

    def __init__(self, max_iters=100, alpha=0.01, print_cost=False):
        self.max_iters = max_iters
        self.alpha = alpha
        self.print_cost = print_cost

    def initialize_with_zeros(self, dim):
        """
        This function creates a vector of zeros of shape (dim, 1)
        for w and initializes b to 0.
        Argument:
        dim -- size of the w vector we want (or number of parameters in this case)
        Returns:
        w -- initialized vector of shape (dim, 1)
        b -- initialized scalar (corresponds to the bias)
        """
        w = np.zeros((dim, 1))
        b = 0
        assert w.shape == (dim, 1)
        assert isinstance(b, (float, int))
        return w, b

    def propagate(self, w, b, X, Y):
        """
        Implement the cost function and its gradient for the propagation explained above

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

        Return:
        cost -- negative log-likelihood cost for logistic regression
        dw -- gradient of the loss with respect to w, thus same shape as w
        db -- gradient of the loss with respect to b, thus same shape as b

        Tips:
        - Write your code step by step for the propagation. np.log(), np.dot()
        """
        m = X.shape[1]  # m_samples

        A = sigmoid(np.dot(w.T, X) + b)  # compute activation
        cost = -1.0 / m * (
            np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)))  # compute cost
        # backward propagation
        dw = 1.0 / m * (np.dot(X, (A - Y).T))
        db = 1.0 / m * (np.sum(A - Y))

        assert dw.shape == w.shape
        assert db.dtype == float
        cost = np.squeeze(cost)
        assert cost.shape == ()
        grads = {"dw": dw, "db": db}

        return grads, cost

    def optimize(self, w, b, X, Y):
        """
        This function optimizes w and b by running a gradient descent algorithm

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of shape (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat),
        of shape (1, number of examples)
        max_iters -- number of iterations of the optimization loop
        alpha -- learning rate of the gradient descent update rule
        print_cost -- True to print the loss every 100 steps

        Returns:
        params -- dictionary containing the weights w and bias b
        grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
        costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
        """

        costs = []
        for i in range(self.max_iters):
            grads, cost = self.propagate(w, b, X, Y)
            dw = grads["dw"]
            db = grads["db"]
            # update w, b
            w = w - self.alpha * dw
            b = b - self.alpha * db
            # Record the costs
            if i % 100 == 0:
                costs.append(cost)
            # print
            if self.print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))

        params = {"w": w, "b": b}
        grads = {"dw": dw, "db": db}
        return params, grads, costs

    def fit(self, X_train, Y_train, X_test, Y_test):
        w, b = self.initialize_with_zeros(X_train.shape[0])
        parameters, grads, costs = self.optimize(w, b, X_train, Y_train)
        w = parameters["w"]
        b = parameters["b"]
        Y_prediction_test = self.predict(w, b, X_test)
        Y_prediction_train = self.predict(w, b, X_train)

        # print errors
        print("train accuracy: {} %".format(
            100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(
            100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

        d = {"costs": costs,
             "Y_prediction_test": Y_prediction_test,
             "Y_prediction_train": Y_prediction_train,
             "w": w,
             "b": b,
             "alpha": self.alpha,
             "max_iters": self.max_iters}

        return d

    def predict(self, w, b, X):
        '''
        Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)

        Returns:
        Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
        '''
        m = X.shape[1]
        Y_prediction = np.zeros((1, m))
        w = w.reshape(X.shape[0], 1)

        A = sigmoid(np.dot(w.T, X) + b)

        for i in range(A.shape[1]):  # 或者用生成器表达式做
            if (A[0, i] > 0.5):
                Y_prediction[0][i] = 1
            else:
                Y_prediction[0][i] = 0

        assert Y_prediction.shape == (1, m)
        return Y_prediction


if __name__ == '__main__':
    from lr_utils import load_dataset
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    train_set_x_flatten = train_set_x_orig.reshape(
        train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(
        test_set_x_orig.shape[0], -1).T
    train_set_x = train_set_x_flatten / 255.
    test_set_x = test_set_x_flatten / 255.

    clf = LRmodel(max_iters=2000, alpha=0.005, print_cost=True)
    clf.fit(train_set_x, train_set_y, test_set_x, test_set_y)
