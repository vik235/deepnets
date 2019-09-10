# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 17:10:28 2019

@author: vigupta
"""

# Basic calculations with tensorflow 

import tensorflow as tf
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

learning_rate = 0.01 
training_epochs = 2000
optimizer_ctor = tf.train.GradientDescentOptimizer
display_step = 50 

data = np.matrix(pd.read_csv("linreg-multi-synthetic-2.csv", header=None).values)

#transpose just so that we get a matrix of shape nxm, m : # of samples 
train_X = data[:,0:2].T
train_Y = data[:,2].T
print(train_X.shape)
print(train_Y.shape)

#dimensions 
n=train_X.shape[0]
m=train_X.shape[1]
#lets work on the computation graph: 

X = tf.placeholder(shape = (n, None), name = "X", dtype = "float32")
Y = tf.placeholder(shape = (1, None), name="Y", dtype = "float32")
#The training variables 
#Wt1 = tf.get_variable(name = "Wt1", shape = (1, n))
#bias1 = tf.get_variable(name = "bias1", shape = ())

A= tf.add(tf.matmul(Wt1, X), bias1)

cost = tf.reduce_sum(tf.pow(A-Y, 2))/(2*m)
optimizer = optimizer_ctor(learning_rate).minimize(cost)
training_costs=[]

log_name = "%g, %s" % (learning_rate, optimizer_ctor.__name__)
tf.summary.scalar('C', cost)
summary_node = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(log_name)
print("Open log file with tensorboard")

#Run the initiliazer first 
init = tf.global_variables_initializer()

with tf.Session() as sess: 
    sess.run(init)
    print("Starting, W" ,sess.run(Wt1))
    
    for epoch in range(training_epochs):
        c = sess.run(optimizer, feed_dict = {X:train_X, Y:train_Y})
        #summary_writer.add_summary(c)
        
        if epoch % display_step == 0:
            c = sess.run(cost, feed_dict = {X:train_X, Y:train_Y})
            training_costs.append(c)
            print ("Epoch:", '%04d' % (epoch), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(Wt1), "b=", sess.run(bias1))
    print ("Epochs exhausted. Quitting the optimization !")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})        
    print ("Training cost=", training_cost, "W=", sess.run(Wt1), "b=", sess.run(bias1), '\n')
   
plt.plot(training_costs)
plt.show()