import numpy as np
import matplotlib.pyplot as plt
import math
import time
def sigmoid(Z):    
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache


def relu(Z):
    A = np.maximum(0.00000001*Z, Z)
    cache = Z
    return A, cache

def maxout(Z):
    A = np.max(Z, axis = 0 ,keepdims=True)
    cache = Z
    return A, cache

def tanh(Z):
    A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
    cache = Z
    return A, cache
    
def linear(Z):
    A = Z
    cache = Z
    return A, cache

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0.00000001
    return dZ

def sigmoid_backward(dA, cache):
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ

def maxout_backward(dA, cache):
    return dA
def tanh_backward(dA,cache):
    Z = cache
    s = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
    dZ = dA * (1 - s * s)
    return dZ

def linearm_backward(dA,cache):
    dZ = dA
    
    return dZ
def initialize_parameters_deep(layer_dims):
    np.random.seed(int(time.time()))
    parameters = {}
    parasetremember = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parasetremember["W" + str(l)] = np.zeros((layer_dims[l], layer_dims[l - 1]))
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
        parasetremember["b" + str(l)] = np.zeros((layer_dims[l], 1))

        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters,parasetremember


def linear_forward(A, W, b):
    Z = None
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    elif activation == "maxout":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = maxout(Z)

    elif activation == "linear":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = linear(Z)

    elif activation == "tanh":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = tanh(Z)

    cache = (linear_cache, activation_cache)
    return A, cache


def predict( X, parameters):
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)],
                                             activation="tanh")
    result, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation="linear")
    return result

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev,parameters["W"+str(l)],parameters["b"+str(l)],activation="tanh")
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation="linear")
    caches.append(cache)

    assert (AL.shape == (1, X.shape[1]))

    return AL, caches


def compute_cost(AL, Y):

    m = Y.shape[1]
    cost = None
    # log cost function
    # cost = -np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL))) / m
    cost = 0.5 / m * np.sum((Y-AL)*(Y-AL))
    cost = np.squeeze(cost)
    assert (cost.shape == ())
    return cost


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1. / m * np.dot(dZ, A_prev.T)
    db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):

    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "maxout":
        dZ = maxout_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "tanh":
        dZ = tanh_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "linear":
        dZ = linearm_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = - (Y-AL)
    #for log cost function
    # dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L - 1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache,
                                                                                                  activation="linear")
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, "tanh")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate,parasetremember):
    L = len(parameters) // 2

    for l in range(L):
        parasetremember["W" + str(l+1)]= 0.9 * parasetremember["W" + str(l+1)] +0.1 * grads["dW" + str(l + 1)]
        parasetremember["b" + str(l+1)] = 0.9 *  parasetremember["b" + str(l+1)] +0.1 * grads["db" + str(l + 1)]
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - (learning_rate * parasetremember["W" + str(l+1)])
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - (learning_rate *  parasetremember["b" + str(l+1)])
    return parameters,parasetremember

if __name__ == '__main__':
    np.random.seed(int(time.time()))
    # X = np.random.rand(1,9)*360
    # Y = np.sin(math.pi/180*X)
    X = np.array([[0],[30],[90],[120],[180],[240],[280],[320],[360]]).T
    Y = np.array([[0],[0.5],[1.0],[0.8660254038],[0],[-0.8660254038],[-0.984807753],[-0.6427876097],[0]]).T

    layer_dims = (1,128,1)
    parasetremember , para = initialize_parameters_deep(layer_dims)
    iter_times=[]
    costs =[]
    cost = 100
    cost2 = 100
    studyrate = 0.075
    i = 1
    while cost > 0.001:
        AL, caches = L_model_forward(X,para)
        cost = compute_cost(AL, Y)
        if cost2 - cost < 0.00001:
            studyrate = studyrate*(1+0.000015)
        if cost - cost2 > 0 and studyrate > 0.0000075:
            studyrate = studyrate/(1+0.003)
        cost2 = cost
        costs.append(cost)
        iter_times.append(i)
        grads = L_model_backward(AL,Y,caches)
        para, parasetremember = update_parameters(para, grads, studyrate , parasetremember)
        print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        i = i+1

    plt.plot(iter_times, costs, '-b')
    plt.xlabel("iteration times")
    plt.ylabel("cost")
    plt.grid()
    plt.draw()
    plt.show()
    plt.close()

    X = np.random.rand(1, 361)*360
    Y = np.sin(math.pi/180*X)
    plt.plot(X, Y, 'bo')
    plt.plot(X, predict(X , para), 'ro')
    plt.xlabel("degree")
    plt.ylabel("y=sin(x)")
    plt.grid()
    plt.draw()
    plt.show()
    plt.close()


