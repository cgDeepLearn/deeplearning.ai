# - *-coding: utf-8 -*-
"""
Created on Mon Sep 11 10:18:23 2017
Following the course of Andrew.Ng, DeepLearning I
Second step a multi-layer neuron nets
@author: cgDeepLearn
"""


import numpy as np
import matplotlib.pyplot as plt
import h5py


def sigmoid(Z):
    """
    sigmoid
    """
    A = 1.0 / (1.0 + np.exp(-Z))
    cache = Z
    return A, cache


def sigmoid_backward(dA, cache):
    """sigmoid backward"""
    Z = cache
    s = 1.0 / (1.0 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    assert dZ.shape == Z.shape
    return dZ


def relu(Z):
    """relu"""
    A = np.maximum(0, Z)
    assert A.shape == Z.shape
    cache = Z
    return A, cache


def relu_backward(dA, cache):
    """relu backward
    0 or 1 * dA
    """
    Z = cache
    dZ = np.array(dA, copy=True)  # just convert dz to a correct object
    # when z <=0, set dz =0
    dZ[Z <= 0] = 0
    return dZ


def tanh(Z):
    """tanh"""
    cache = Z
    A = np.tanh(Z)  # A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
    return A, cache


def tanh_backward(dA, cache):
    """tanh backward
    1.0 - x**2
    """
    Z = cache
    A = np.tanh(Z)
    dZ = dA * (1.0 - A * A)
    return dZ


activations = {"sigmoid": (sigmoid, sigmoid_backward),
               "relu": (relu, relu_backward),
               "tanh": (tanh, tanh_backward)
               }


class NeuralLayer():
    """
    nn layer
    """

    def __init__(self, name, W, b, num_units, activate_function):
        self.name = name
        self.activation = activate_function
        self.num_units = num_units
        self.W = W
        self.b = b


class NetStruct():
    """神经网络结构"""

    def __init__(self, layer_dims, activate_func_list):
        """
        layer_dims: [x_input , *hidden_layer_dims, y_output]
        activate_func_list : ['sigmoid','relu', 'linear', 'tanh'...]
        """
        self.layer_dims = layer_dims  # 各层神经元数目
        self.activate_func_list = activate_func_list
        # 搭建神经层
        self.layers = []
        self.params = {}
        np.random.seed(1)
        for l in range(1, len(self.layer_dims)):
            self.params['W' + str(l)] = np.random.randn(
                self.layer_dims[l], self.layer_dims[l - 1]) / np.sqrt(self.layer_dims[l - 1])
            self.params['b' + str(l)] = np.zeros((self.layer_dims[l], 1))
            self.layers.append(NeuralLayer("L" + str(l),
                                           self.params['W' + str(l)],
                                           self.params['b' + str(l)],
                                           self.layer_dims[l],
                                           self.activate_func_list[l - 1]))


class NNModel():
    """
    nn model
    """

    def __init__(self, net_struct, alpha=0.01, iterations=1000, print_cost=False):
        """初始化"""
        self.alpha = alpha
        self.iterations = iterations
        self.L = len(net_struct.layer_dims) - 1  # L layer nn
        self.layers = net_struct.layers
        self.params = net_struct.params
        self.print_cost = print_cost

    def linear_forward(self, A, W, b):
        """linear前向传播"""
        Z = np.dot(W, A) + b
        assert Z.shape == (W.shape[0], A.shape[1])
        cache = (A, W, b)

        return Z, cache

    def linear_activation_forward(self, A_prev, W, b, activation):
        """激活函数后的结果"""
        activate_func = activations[activation][0]
        Z, linear_cache = self.linear_forward(A_prev, W, b)
        A, activation_cache = activate_func(Z)
        cache = (linear_cache, activation_cache)
        return A, cache

    def L_model_forward(self, X):
        """L layers forward
        X:input
        AL:last activation value
        """
        caches = []  # list of caches (Z)
        A = X
        L = len(self.layers)
        layers = self.layers

        for l in range(1, L + 1):
            A_prev = A
            W = self.params["W" + str(l)]
            b = self.params["b" + str(l)]
            activation = layers[l - 1].activation
            A, cache = self.linear_activation_forward(A_prev, W, b, activation)
            caches.append(cache)
        AL = A
        assert AL.shape == (1, X.shape[1])

        return AL, caches

    def compute_cost(self, AL, Y):
        """compute cost"""
        m = Y.shape[1]  # sample size
        cost = -1. / m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))

        # To make sure your cost's shape is what we expect (e.g. this turns
        # [[17]] into 17).
        cost = np.squeeze(cost)
        assert cost.shape == ()

        return cost

    def linear_backward(self, dZ, cache):
        """linear backward"""
        A_prev, W, b = cache
        m = A_prev.shape[1]
        dW = 1. / m * np.dot(dZ, A_prev.T)
        db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        assert dA_prev.shape == A_prev.shape
        assert dW.shape == W.shape

        return dA_prev, dW, db

    def linear_activation_backward(self, dA, cache, activation):
        """single layer activation backward"""
        linear_cache, activation_cache = cache
        activation_backward_func = activations[activation][1]
        dZ = activation_backward_func(dA, activation_cache)
        dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        return dA_prev, dW, db

    def L_model_backward(self, AL, Y, caches):
        """L layer backward"""
        grads = {}
        L = len(caches)  # layers num
        Y = Y.reshape(AL.shape)

        dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        grads["dA" + str(L + 1)] = dAL  # 计算dAn时用

        # 计算dAl
        for l in reversed(range(1, L + 1)):
            current_cache = caches[l - 1]
            temp_dA, temp_dW, temp_db = self.linear_activation_backward(
                grads["dA" + str(l + 1)],
                current_cache,
                self.layers[l - 1].activation)
            grads["dA" + str(l)] = temp_dA
            grads["dW" + str(l)] = temp_dW
            grads["db" + str(l)] = temp_db
        return grads

    def update_params(self, grads):
        """ update params
        w = w - alpha* grad
        """
        L = len(self.layers)
        for l in range(1, L + 1):
            self.params["W" + str(l)] -= self.alpha * grads["dW" + str(l)]
            self.params["b" + str(l)] -= self.alpha * grads["db" + str(l)]

    def train(self, X, Y):
        grads = {}
        costs = []
        # Loop
        for i in range(self.iterations):
            AL, caches = self.L_model_forward(X)
            cost = self.compute_cost(AL, Y)

            grads = self.L_model_backward(AL, Y, caches)
            self.update_params(grads)

            if i % 100 == 0:
                costs.append(cost)
                if self.print_cost:
                    print("Cost after itration %i: %f" % (i, cost))

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("learning rate = " + str(self.alpha))
        plt.show()

        return self.params

    def predict(self, X, y):
        """
        predict after train
        """
        m = X.shape[1]
        predictions = np.zeros((1, m))
        probas, _ = self.L_model_forward(X)

        # convert probas to 0/1 predictions
        for i in range(probas.shape[1]):  # probas.shape = (1, m)
            if probas[0, i] > 0.5:
                predictions[0, i] = 1
            else:
                predictions[0, i] = 0
        print("Accuracy:" + str(np.sum((predictions == y) / m)))

        return predictions


def load_data():
    """load data """
    with h5py.File('datasets/train_catvnoncat.h5', "r") as train_dataset:

        train_set_x_orig = np.array(train_dataset["train_set_x"][
            :])  # your train set features
        train_set_y_orig = np.array(train_dataset["train_set_y"][
            :])  # your train set labels

        test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
        test_set_x_orig = np.array(test_dataset["test_set_x"][:])
        test_set_y_orig = np.array(test_dataset["test_set_y"][:])

        classes = np.array(test_dataset["list_classes"][:])

        train_set_y_orig = train_set_y_orig.reshape(
            (1, train_set_y_orig.shape[0]))
        test_set_y_orig = test_set_y_orig.reshape(
            (1, test_set_y_orig.shape[0]))
        return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

if __name__ == '__main__':
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
    # The "-1" makes reshape flatten the remaining dimensions
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten / 255.
    test_x = test_x_flatten / 255.
    np.random.seed(1)
    # 确定神经网络结构
    # layer dims ,firt->input,last->output ,mid->hidden layer dims
    layer_dims = [12288, 20, 7, 5, 1]
    # activate_functions for each layer,len= len(dims) - 1
    activate_functions = ["relu", "relu", "relu", "sigmoid"]
    ns = NetStruct(layer_dims, activate_functions)
    model = NNModel(ns, alpha=0.0075, iterations=2500, print_cost=True)
    # print("w1:", model.params["W1"][:, 0])
    params = model.train(train_x, train_y)
    preditions = model.predict(test_x, test_y)
