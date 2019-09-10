# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Basic calculations with tensorflow 

import tensorflow as tf

a = tf.constant(10, dtype = float)
b = tf.constant(20, dtype = float)

with tf.Session() as sess: 
    print('The result of adding two constant tensors %i and %i is  %i ' % (sess.run(a), sess.run(b), sess.run(a+b)))
    print('The result of multiplying two constant tensors %i and %i is  %i ' % (sess.run(a), sess.run(b), sess.run(tf.multiply(a, b))))
    print('The result of substracting two constant tensors %i and %i is  %i ' % (sess.run(a), sess.run(b), sess.run(tf.subtract(a, b))))
    
# Working with the tf computaional graph

#Define  some placeholder 
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

#Define some neurons (with basic functions on them)

add = tf.add(a, b)
sub = tf.subtract(a, b)
mul = tf.multiply(a, b)
add2 = tf.add(add, sub)
#Run the graph 
with tf.Session() as sess:
    print ("Running a chained, nested operation/graph %i" % sess.run(add2, feed_dict = {a:20, b:30} ))
    print ("Running addition %i" % sess.run(add2, feed_dict = {a:20, b:30} ))
    
    
#Matrix operations:
matrix1 = tf.constant([[3., 3.]], name="matrix1")    
matrix2 = tf.constant([[3.],[3.]], name="matrix2")    
print(matrix1)
with tf.Session() as sess:
    print("Matrix 1 is " , sess.run(matrix1))
    print("Matrix 2 is " , sess.run(matrix2))
    print("Matrix Mult is ",  sess.run(tf.matmul(matrix1, matrix2)))