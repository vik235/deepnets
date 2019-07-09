# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 04:24:49 2019
@author: vigupta
helper functions for neuralnets. 
"""

import numpy as np
import h5py
    
    
def load_dataset():
     """
    this loads a h5 dataset.
    courtesy coursera 
    specialized for catvsnotcat
    """
    train_dataset = h5py.File('C:/Users/vigupta/source/repos/deepnets/Shallownets/datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) #  train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) #  train set labels

    test_dataset = h5py.File('C:/Users/vigupta/source/repos/deepnets/Shallownets/datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def sigmoid(z):
     """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """
    