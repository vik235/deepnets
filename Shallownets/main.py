# -*- coding: utf-8 -*-
"""
Loading packages for the script 

1. [numpy]: Needed for doing computing , python fundamentals 
2. [h5py]: This is a common package to interact with a dataset that is stored on an H5 file.
3. [matplotlib]: Plotting libraries
4. [PIL]/scipy: Testing the model
"""

import numpy as np 
import matplotlib.pyplot as plt 
import h5py
import scipy 
from PIL import Image
from scipy import ndimage
from dsutils import load_dataset 



train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

"""
loading an image and checking 
"""

index = 1
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:, index]) + 
       ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") + 
       "' picture.")

m_train = train_set_y.shape[1]
m_test =  test_set_y.shape[1]
num_px = train_set_x_orig.shape[1]
    
print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

"""
Flattening the imaages from RGB channel to a flattened X which of shape n_x, m : n_x being the stacked up channels 
and also the dimensions of the input features 

Note the trick to flatten:
    
A trick when you want to flatten a matrix X of shape (a,b,c,d) to a matrix X_flatten of shape (b ∗∗ c ∗∗ d, a) is to use:
X_flatten = X.reshape(X.shape[0], -1).T

"""
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0] , -1).T

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))


"""
Standardizing the input . 255 is the max value of the channel
"""

train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

