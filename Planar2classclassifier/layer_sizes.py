# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 00:34:29 2019

@author: vigupta
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
    
    W1 = np.random.randn((n_h, n_x)) * 0.01 
    b1 = 0
    
    W2 = np.random.randn((n_y, n_h)) * 0.01 
    b2 = 0
    
    parameters = {"W1" : W1,
                  "b1" : b1,
                  "W2" : W2,
                  "b2" : b2
                  }
    
    assert(parameters["W1"].shape == (n_h, n_x))
    assert(parameters["W2"].shape == (n_y, n_h))