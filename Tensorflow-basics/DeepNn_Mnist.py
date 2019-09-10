# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 00:34:19 2019

@author: vigupta
"""

from __future__ import print_function 
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('/mnsit/data', one_hot=True)

#HParameters of the model 

learning_rate=0.01 
training_epochs=1000
batch_size=128
display_step=100

#network architectute 


n_hidden1=256 
n_hidden2=520 
n_input=784
num_classes=10

#building the graph 
X = tf.placeholder(dtype="float32", shape=(None, n_input), name="X")
Y = tf.placeholder(dtype="float32", shape=(None, num_classes), name="Y")

Weights= {
                "h1":tf.Variable(tf.random_normal((n_input, n_hidden1)), name="h1"),
                "h2":tf.Variable(tf.random_normal((n_hidden1, n_hidden2)), name="h1"),
                "out":tf.Variable(tf.random_normal((n_hidden2, num_classes)), name="out")
        }

Biases = {
         "b1":tf.Variable(tf.random_normal([n_hidden1]), name="b1"),
         "b2":tf.Variable(tf.random_normal([n_hidden2]), name="b2"),
         "out":tf.Variable(tf.random_normal([num_classes]), name="out")
        }


def neural_net(X):
    layer1 = tf.add(tf.matmul(X, Weights["h1"]), Biases["b1"])
    layer2 = tf.add(tf.matmul(layer1, Weights["h2"]), Biases["b2"])
    out = tf.add(tf.matmul(layer2, Weights["out"]), Biases["out"])
    return out

logits = neural_net(X)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
training_losses = []
training_accuracy = []
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={X:batch_x, Y:batch_y})
        if(epoch%display_step == 0 ):
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X:batch_x, Y:batch_y})
            training_losses.append(loss)
            training_accuracy.append(acc)
            print("Step " + str(epoch) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
    print ("Epochs exhausted. Quitting the optimization !")
plt.plot(training_losses)        
plt.show()
plt.plot(training_accuracy)        
plt.show()
        
