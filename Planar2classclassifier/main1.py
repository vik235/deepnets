# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 00:09:05 2019
@author: vigupta
Main worker for the 2 class classifier. We will just manually 
build a hidden layer for some fun with matmul and get soem intuition


"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn # to try out classical Ml model. 
import sklearn.datasets 
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
from nn_arch_helpers import nn_model

"""
Set the seed value for consistent results 
"""
np.random.seed(1)

"""
Load the dataset 
"""

X, Y = load_planar_dataset()
print(X.shape)
print(Y.shape)

print(np.squeeze(Y))
print(np.squeeze(X[:,1]))

plt.scatter(X[0, :], X[1, :], s=40, c = np.squeeze(Y), cmap=plt.cm.Spectral);

m = X.shape[1]

"""
Develop a neural network , shallow of n_h = 4 , and L = 2 , output layer has activation of sigmoid and activation 
for l = 1 is tanh. 

Cost function J(W , b, X, Y) = -(1/m)*np.sum(np.multiply(Y, logA2) + (I - Y)*log(I - A2))
"""
n_x, n_y, n_h = layer_sizes(X, Y)
print("Input layer size, dims: " + str(n_x) +", Hidden layer size: " + str(n_h) + ", Outpur layer dim: " +str(n_y) )

parameters = nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost = True) 






