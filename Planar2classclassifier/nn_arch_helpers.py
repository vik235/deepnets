# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 00:34:29 2019

@author: vigupta

nn architecture helper 
"""
import numpy as np 
import sklearn 

def layer_sizes(X, Y):
    """
    Here we build really a shallow neural net with a 4 unit single hidden layer 
    n_x and n_y i.e. the size of the input and the output layer needs to be inferred from input vars 
    n_h is 4 
    
    X, Y: X is the input dataset(n_x, m) , Y is the labels, (n_y, m )
    m is the trainign examples in above 
    
    returns n_x, n_h , n_ysizes of the input, output and the hidden layer
    """
    return X.shape[0], 4, Y.shape[0] 

def initialize_parameters(n_x, n_h, n_y):
    """
    n_x and n_y i.e. the size of the input and the output layer needs to be inferred from input vars 
    n_h is 4 
    
    returns a dict with initilized params 
    W1 : Weight matrix for the first layer. (n_h, n_x)
    b1: Bias vector for layer 1. (n_h, 1)
    W2 : Weight matrix for the second layer. (n_y, n_h)
    b2: Bias vector for layer 2. (n_y, 1)
    
    """
    np.random.seed(2)
    
    W1 = np.random.randn(n_h, n_x) * 0.01 
    b1 = 0
    
    W2 = np.random.randn(n_y, n_h) * 0.01 
    b2 = 0
    
    parameters = {"W1" : W1,
                  "b1" : b1,
                  "W2" : W2,
                  "b2" : b2
                  }
    
    assert(parameters["W1"].shape == (n_h, n_x))
    assert(parameters["W2"].shape == (n_y, n_h))
    
    return parameters

def sigmoid(x):
    """
    Compute the sigmoid of x
    Arguments:
    x -- A scalar or numpy array of any size.
    Return:
    s -- sigmoid(x)
    """
    s = 1/(1+np.exp(-x))
    return s

def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    
    Z1 = np.dot(parameters["W1"],X) + parameters["b1"]
    A1 = np.tanh(Z1)
    Z2 = np.dot(parameters["W2"],A1) + parameters["b2"]
    A2 = sigmoid(Z2)
    
    cache = {"Z1":Z1,
             "A1":A1,
             "Z2":Z2,
             "A2":A2
            }
    
    return A2, cache

def compute_cost(A2, Y):
    """
    A2 : final activations , 
    Y: true values 
    
    returns cost --- categorical cross entropy given by deviance. 
    """
    m = Y.shape[1]

    logprobs = np.multiply(np.log(A2),Y) + np.multiply(np.log(1 - A2),(1 - Y))
    cost = - (1/m)*np.sum(logprobs)
    
    
    cost = float(np.squeeze(cost))  
    assert(isinstance(cost, float))
    return cost
    
def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using 
    Arguments:
    parameters -- python dictionary containing our parameters 
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    
    Returns:
    grads -- python dictionary containing the gradients with respect to different parameters
    """
    m = X.shape[1]
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    dZ2 = A2 - Y
    dW2 = (1/m)*np.dot(dZ2, A1.T)
    db2 = (1/m)*np.sum(dZ2, axis = 1, keepdims=True)
    dZ1 = (np.dot(W2.T,dZ2))*(1 - np.power(A1,2))
    dW1 = (1/m)*np.dot(dZ1,X.T)
    db1 = (1/m)*np.sum(dZ1, axis = 1, keepdims=True)
    
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads    
    

def update_parameters(parameters, grads, learning_rate = 1.2):
    """
    Updates parameters using the gradient descent . Deos a single pass of grad descent and returns the updated parameters 
    Note: learnign rate can be a hyper parameter to tune
    
    Arguments:
    parameters -- python dictionary containing parameters 
    grads -- python dictionary containing gradients 
    
    Returns:
    parameters -- python dictionary containing updated parameters 
    """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def nn_model(X, Y, n_h, num_iterations = 10000, print_cost = False):
    np.random.seed(3)
    n_x, n_h, n_y = layer_sizes(X, Y)
    parameters = initialize_parameters(n_x, n_h, n_y)
    costs = []
    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate= 1.2)
        costs.append(cost)
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters
        
def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    predictions  = np.where(A2 > 0.5, 1, 0)
    
    return predictions
        