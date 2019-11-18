# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 17:15:02 2019

@author: pc
"""

import tensorflow as tf

xavier = tf.contrib.layers.xavier_initializer()
msra   = tf.contrib.layers.variance_scaling_initializer()
'''
如果使用 relu，则最好使用 he_initialization, 即 tf.contrib.layers.variance_scaling_initializer( )
在 relu 网络中，假定每一层有一半的神经元被激活，另一半为 0 ，所以要保持 variance 不变，只需要在 Xavier 的基础上再除以 2 。
tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.variance_scaling_initializer())

如果激活函数使用 sigmoid 和 tanh，则最好使用 xavier initialization， 即 tf.contrib.layers.xavier_initializer_conv2d( )
xavier 是保持输入和输出的方差一致，避免了所有的输出值都趋向于0.
tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
'''

def Variable_with_weight_loss(shape, name, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape,stddev=stddev), name=name)
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var),wl,name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var

def Variable_with_weight_loss_xavier(shape, name , wl):
    var = tf.get_variable(name=name,shape=shape,initializer=xavier)
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var),wl,name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var

def Variable_with_weight_loss_msra(shape, name , wl):
    var = tf.get_variable(name=name,shape=shape,initializer=msra)
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var),wl,name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var

def Conv(in_op, name, size, step, n_out, bias=0.0, init='', wl=0.0, padding='SAME'):
    n_in = in_op.get_shape()[-1].value
    with tf.variable_scope(name):
        shape   = [size, size, n_in, n_out]
        if init == 'xavier':
            weights = Variable_with_weight_loss_xavier(shape, 'conv_w', wl=wl)
        elif init == 'msra':
            weights = Variable_with_weight_loss_msra(shape, 'conv_w', wl=wl)
        else:
            weights = Variable_with_weight_loss(shape, 'conv_w', stddev=0.1, wl=0)
        biases  = tf.Variable(tf.constant(bias,shape=[n_out]),name='biases')
        conv    = tf.nn.conv2d(in_op, weights, strides=[1,step,step,1],padding=padding)
        relu    = tf.nn.relu(tf.nn.bias_add(conv, biases))
    return relu

def LRN(in_op,name='lrn'):
    with tf.name_scope(name):
        norm = tf.nn.lrn(in_op, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
    return norm

def Maxpool(in_op,name,size,step, padding='SAME'):
    with tf.variable_scope(name):
        pool = tf.nn.max_pool(in_op,ksize=[1,size,size,1], strides=[1,step,step,1],padding=padding)
    return pool

def Flatten(in_op):
    shape      = in_op.get_shape().as_list()
    nodes      = shape[1]*shape[2]*shape[3]
    reshaped   = tf.reshape(in_op,[-1, nodes])
    return reshaped

def Fc(in_op, name, n_out, bias=0.1, init='', wl=0.0, activation=None):
    n_in = in_op.get_shape()[-1].value
    with tf.variable_scope(name):
        shape = [n_in,n_out]
        if init == 'xavier':
            fc_weights = Variable_with_weight_loss_xavier(shape, 'fc_w', wl=wl)
        elif init == 'msra':
            fc_weights = Variable_with_weight_loss_msra(shape, 'fc_w', wl=wl)
        else:
            fc_weights = Variable_with_weight_loss(shape, 'fc_w', stddev=0.1, wl=wl)
        fc_biases  = tf.Variable(tf.constant(bias, shape=[n_out]), name='biases')
        if activation == 'relu':
            fc         = tf.nn.relu(tf.matmul(in_op, fc_weights) + fc_biases)
        elif activation == 'sfotmax':
            fc         = tf.nn.softmax(tf.matmul(in_op, fc_weights) + fc_biases)
        else:
            fc         = tf.matmul(in_op, fc_weights) + fc_biases
    return fc