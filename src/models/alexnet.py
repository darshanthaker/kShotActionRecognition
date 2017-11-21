import tensorflow as tf
from pdb import set_trace
from util import *
import numpy as np
import scipy.io

def conv_layer(input, num_outputs, kernel_size, stride=1):
    return tf.contrib.layers.conv2d(input, num_outputs=num_outputs,
        kernel_size=kernel_size, stride=stride, weights_regularizer=tf.nn.l2_loss)

def pool_layer(input):
    return tf.nn.max_pool(input, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1),
            padding='SAME')

def fc_layer(input, n, activation_fn=tf.nn.relu):
    return tf.contrib.layers.fully_connected(input, n, activation_fn=activation_fn, \
        weights_regularizer=tf.nn.l2_loss)  

def lrn_layer(input):
    depth_radius = 5
    alpha = 1e-04
    beta = 0.75
    bias = 2.0
    return tf.nn.local_response_normalization(input, depth_radius=depth_radius, \
        alpha=alpha, beta=beta, bias=bias)

class AlexNet(object):

    def __init__(self):
        pass

    def feed_forward(self, input_image, scope='AlexNet', architecture='mynet1'):
        net = {}

        if architecture == 'alexnet':
            with tf.variable_scope(scope + '/' + architecture):
                net['conv_1'] = conv_layer(input_image, 64, 11, stride=4)
                net['conv_1'] = lrn_layer(net['conv_1'])
                net['pool_1'] = pool_layer(net['conv_1'])
                net['conv_2'] = conv_layer(net['pool_1'], 192, 5)
                net['conv_2'] = lrn_layer(net['conv_2'])
                net['pool_2'] = pool_layer(net['conv_2'])
                net['conv_3'] = conv_layer(net['pool_2'], 384, 3)
                net['conv_4'] = conv_layer(net['conv_3'], 384, 3)
                net['conv_5'] = conv_layer(net['conv_4'], 256, 3)
                net['pool_3'] = pool_layer(net['conv_5'])

                net['pool_3'] = tf.reshape(net['pool_3'], [-1, 16384])
            
                net['fc_1'] = fc_layer(net['pool_3'], 128)
                net['fc_2'] = fc_layer(net['fc_1'], 128)
                net['output'] = fc_layer(net['fc_2'], NUM_CLASSES)
        elif architecture == 'mynet1':
            with tf.variable_scope(scope + '/' + architecture):
                net['conv_1'] = conv_layer(input_image, 32, 3)
                net['conv_1'] = tf.layers.batch_normalization(net['conv_1'])
                net['pool_1'] = pool_layer(net['conv_1'])
                net['conv_2'] = conv_layer(net['pool_1'], 32, 3)
                net['conv_2'] = tf.layers.batch_normalization(net['conv_2'])
                net['pool_2'] = pool_layer(net['conv_2'])
                net['conv_3'] = conv_layer(net['pool_2'], 32, 3)
                net['conv_3'] = tf.layers.batch_normalization(net['conv_3'])
                net['conv_4'] = conv_layer(net['conv_3'], 64, 3)
                net['conv_4'] = tf.layers.batch_normalization(net['conv_4'])
                net['pool_5'] = pool_layer(net['conv_4'])
                #net['conv_5'] = conv_layer(net['conv_4'], 32, 3)
                #net['conv_5'] = tf.layers.batch_normalization(net['conv_5'])
                #net['pool_5'] = pool_layer(net['conv_5'])
                #net['conv_6'] = conv_layer(net['pool_3'], 64, 3)
                #net['conv_6'] = tf.layers.batch_normalization(net['conv_6'])
                #net['pool_4'] = pool_layer(net['conv_6'])
                #net['conv_7'] = conv_layer(net['pool_4'], 64, 3)
                #net['conv_7'] = tf.layers.batch_normalization(net['conv_7'])
                #net['pool_5'] = pool_layer(net['conv_7'])

                b, h, w, c = net['pool_5'].shape.as_list() 
                net['pool_5'] = tf.reshape(net['pool_5'], [-1, h*w*c])
            
                net['fc_1'] = fc_layer(net['pool_5'], 64) 
                net['fc_1'] = tf.layers.batch_normalization(net['fc_1'])
                net['fc_1'] = tf.layers.dropout(net['fc_1'], 0.5) 
                net['fc_2'] = fc_layer(net['fc_1'], 64)
                net['fc_2'] = tf.layers.batch_normalization(net['fc_1'])
                net['fc_1'] = tf.layers.dropout(net['fc_1'], 0.5) 
                net['output'] = fc_layer(net['fc_2'], NUM_CLASSES)
        elif architecture == 'mynet2':
            with tf.variable_scope(scope + '/' + architecture):
                net['conv_1'] = conv_layer(input_image, 32, 3)
                net['conv_1'] = tf.layers.batch_normalization(net['conv_1'])
                net['pool_1'] = pool_layer(net['conv_1'])
                net['conv_2'] = conv_layer(net['pool_1'], 32, 3)
                net['conv_2'] = tf.layers.batch_normalization(net['conv_2'])
                net['pool_2'] = pool_layer(net['conv_2'])
                net['conv_3'] = conv_layer(net['pool_2'], 64, 3)
                net['conv_4'] = conv_layer(net['conv_3'], 64, 3)
                net['conv_5'] = conv_layer(net['conv_4'], 64, 3)
                net['pool_3'] = pool_layer(net['conv_5'])

                b, h, w, c = net['pool_3'].shape.as_list() 
                net['pool_3'] = tf.reshape(net['pool_3'], [-1, h*w*c])
            
                net['fc_1'] = fc_layer(net['pool_3'], 64)
                net['fc_1'] = tf.layers.dropout(net['fc_1'], 0.5) 
                net['fc_2'] = fc_layer(net['fc_1'], 64)
                net['fc_1'] = tf.layers.dropout(net['fc_1'], 0.5) 
                net['output'] = fc_layer(net['fc_2'], NUM_CLASSES)
        elif architecture == 'mynet3':
            with tf.variable_scope(scope + '/' + architecture):
                net['conv_1'] = conv_layer(input_image, 32, 7)
                net['conv_1'] = tf.layers.batch_normalization(net['conv_1'])
                net['pool_1'] = pool_layer(net['conv_1'])
                net['conv_2'] = conv_layer(net['pool_1'], 32, 5)
                net['conv_2'] = tf.layers.batch_normalization(net['conv_2'])
                net['pool_2'] = pool_layer(net['conv_2'])
                #net['conv_3'] = conv_layer(net['pool_2'], 64, 3)
                #net['conv_4'] = conv_layer(net['conv_3'], 64, 3)
                #net['conv_5'] = conv_layer(net['conv_4'], 64, 3)
                #net['pool_3'] = pool_layer(net['conv_5'])

                b, h, w, c = net['pool_2'].shape.as_list() 
                net['pool_2'] = tf.reshape(net['pool_2'], [-1, h*w*c])
            
                net['fc_1'] = fc_layer(net['pool_2'], 64)
                net['fc_1'] = tf.layers.batch_normalization(net['fc_1'])
                net['fc_1'] = tf.layers.dropout(net['fc_1'], 0.5) 
                net['fc_2'] = fc_layer(net['fc_1'], 64)
                net['fc_2'] = tf.layers.batch_normalization(net['fc_2'])
                net['fc_2'] = tf.layers.dropout(net['fc_2'], 0.5) 
                net['output'] = fc_layer(net['fc_2'], NUM_CLASSES)
        elif architecture == 'mynet4':
            with tf.variable_scope(scope + '/' + architecture):
                net['conv_1'] = conv_layer(input_image, 64, 3)
                net['conv_1'] = tf.layers.batch_normalization(net['conv_1'])
                net['pool_1'] = pool_layer(net['conv_1'])
                net['conv_2'] = conv_layer(net['pool_1'], 64, 3)
                net['conv_2'] = tf.layers.batch_normalization(net['conv_2'])
                net['pool_2'] = pool_layer(net['conv_2'])
                #net['conv_3'] = conv_layer(net['pool_2'], 64, 3)
                #net['conv_4'] = conv_layer(net['conv_3'], 64, 3)
                #net['conv_5'] = conv_layer(net['conv_4'], 64, 3)
                #net['pool_3'] = pool_layer(net['conv_5'])

                b, h, w, c = net['pool_2'].shape.as_list() 
                net['pool_2'] = tf.reshape(net['pool_2'], [-1, h*w*c])
            
                net['fc_1'] = fc_layer(net['pool_2'], 16)
                net['fc_1'] = tf.layers.dropout(net['fc_1'], 0.5) 
                net['fc_2'] = fc_layer(net['fc_1'], 16)
                net['fc_1'] = tf.layers.dropout(net['fc_1'], 0.5) 
                net['output'] = fc_layer(net['fc_2'], NUM_CLASSES)

        return net
