# - *-coding: utf-8 -*-
"""
Created on Mon Sep 11 10:18:23 2017
Following the course of Andrew.Ng, DeepLearning I
Second step a multi-layer neuron nets
@author: cgDeepLearn
"""


import numpy as np


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
    dZ = dA * (1.0 - Z * Z)
    return dZ


activations = {"sigmoid": (sigmoid, sigmoid_backward),
               "relu": (rel, relu_backward),
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
        for l in range(1, len(self.layer_dims)):
            self.params[
                'W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l - 1]) * 0.01
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

    def __init__(self, net_struct, alpha=0.01, iteration=1000):
        """初始化"""
        self.alpha = alpha
        self.iteration = iteration
        self.layers = self.net_struct.layers
        self.params = self.net_struct.params

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

    def L_model_forward(self, X):
        """L layers forward
        X:input
        AL:last activation value
        """
        caches = []  # list of caches (Z)
        A = X
        L = len(self.layers)
        layers = self.layers

        for l in range(1, L):
            A_prev = A
            A, cache = self.linear_activation_forward(A_prev,
                                                      layers[l - 1].W,
                                                      layers[l - 1].b,
                                                      layers[l - 1].activation)
            caches.append(cache)
        AL = A
        assert AL.shape == (1, X.shape[1])

        return AL, caches

    def comput_cost(self, AL, Y):
        """compute cost"""
        m = Y.shape[1]  # sample size
        cost = -1. / m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))

        # To make sure your cost's shape is what we expect (e.g. this turns
        # [[17]] into 17).
        cost = np.squeeze(cost)
        assert cost.sahpe == ()

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
        activation_backward_func = activations[activation][1]
        dZ = activation_backward_func(dA, cache)
        dA_prev, dW, db = self.linear_backward(dZ, cache)

    def L_model_backward(self, AL, Y, caches):
        """L layer backward"""
        grads = {}
        L = len(caches)  # layers num
        m = Al.shape[1]  # samples num
        Y = Y.reshape(AL.shape)

        dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - Al))
        grads["dA" + str(L + 1)] = dAL  # 计算dAn时用

        # 计算dAl
        for l in reversed(range(1, L + 1)):
            current_cache = caches[l]
        grads["dA" + str(l)], grads["dW" + str(l)], grads["db" + str(l)] = self.linear_activation_backward(grads["dA" + str(l + 1)],
                                                                                                           current_cache,
                                                                                                           self.layers[-1].activation)
        return grads

    def update_params(self, grads):
        """ update params
        w = w - alpha* grad
        """
        L = len(self.layers)
        for l in range(1, L + 1):
            self.params["W" + str(l)] -= self.alpha * grads["dW" + str(l)]
            self.params["b" + str(l)] -= self.alpha * grads["db" + str(l)]

    def train(self, X, Y)
