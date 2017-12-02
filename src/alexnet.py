"""
Adapted from https://github.com/guerzh/tf_weights/blob/master/myalexnet_forward_newtf.py
"""
import tensorflow as tf
from pdb import set_trace
from util import *
import numpy as np
import scipy.io

def conv_layer(input, weights, bias, stride=1, group=1):
    if group == 1:
        conv = tf.nn.conv2d(input, weights, strides=(1, stride, stride, 1),
            padding='SAME')
    else:
        input_groups = tf.split(input, group, 3)
        kernel_groups = tf.split(weights, group, 3)
        output_groups = list()
        for (i, k) in zip(input_groups, kernel_groups):
            output = tf.nn.conv2d(i, k, strides=(1, stride, stride, 1), \
                padding='SAME')
            output_groups.append(output)
        conv = tf.concat(output_groups, 3)
    return tf.reshape(tf.nn.bias_add(conv, bias), \
         [-1]+conv.get_shape().as_list()[1:])

def pool_layer(input):
    return tf.nn.max_pool(input, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1),
            padding='VALID')

def fc_layer(input, n, activation_fn=tf.nn.relu):
    return tf.contrib.layers.fully_connected(input, n, activation_fn=activation_fn, \
        weights_regularizer=tf.nn.l2_loss)  

def lrno_layer(input):
    depth_radius = 5
    alpha = 1e-04
    beta = 0.75
    bias = 2.0
    return tf.nn.local_response_normalization(input, depth_radius=depth_radius, \
        alpha=alpha, beta=beta, bias=bias)

class AlexNet(object):

    layers = (
        'conv1', 'lrno1', 'pool1',
        'conv2', 'lrno2', 'pool2',
        'conv3',
        'conv4',
        'conv5',
    )
    strides = {
        'conv1': 4,
        'conv2': 1,
        'conv3': 1,
        'conv4': 1,
        'conv5': 1
    }
    groups = {
        'conv1': 1,
        'conv2': 2,
        'conv3': 1,
        'conv4': 2,
        'conv5': 2
    }

    def __init__(self):
        eprint("WARNING: Using pretrained Alexnet")
        data_path = 'bvlc_alexnet.npy'
        self.weights = np.load(data_path, encoding='bytes').item()

    def load_weights(self):
        self.net = {}
        for name in self.layers:
            kind = name[:4]
            if kind == 'conv':
                kernels = self.weights[name][0]
                bias = self.weights[name][1]

                self.net['{}_kernel'.format(name)] = tf.Variable(kernels, \
                    name='{}_kernel'.format(name))
                self.net['{}_bias'.format(name)] = tf.Variable(bias, \
                    name='{}_bias'.format(name))

    def feed_forward(self, input_image, scope='AlexNet', architecture='vanilla', 
            num_hidden=2048, dp=0.5):
        net = {}

        self.num_hidden = num_hidden
        self.dp = dp

        current = input_image

        with tf.variable_scope(scope):
            for name in self.layers:
                kind = name[:4] 
                if kind == 'conv':
                    kernels = self.net['{}_kernel'.format(name)]
                    bias = self.net['{}_bias'.format(name)]
                    current = conv_layer(current, kernels, bias, \
                        stride=self.strides[name], group=self.groups[name])
                    current = tf.nn.relu(current)
                elif kind == 'pool':
                    current = pool_layer(current)
                elif kind == 'lrno':
                    current = lrno_layer(current)
                net[name] = current
        net['output'] = net['conv5']
               
        return net
