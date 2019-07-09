# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 05:19:28 2019

@author: vigupta

Contains implemeting various activation functions

"""

import numpy as np

def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """
    s = np.divide(1, 1 + np.exp(-z))
    return s

def relu(z):
    """
    Compute the relu of linear function z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    r -- relu(z)
    """
    r = np.max(0, z)
    
    return r