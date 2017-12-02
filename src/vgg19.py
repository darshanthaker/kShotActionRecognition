import tensorflow as tf
from pdb import set_trace
from util import *
import numpy as np
import scipy.io

def _conv_layer(input, weights, bias):
    conv = tf.nn.conv2d(input, weights, strides=(1, 1, 1, 1),
            padding='SAME')
    return tf.nn.bias_add(conv, bias)

def _pool_layer(input):
    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
            padding='SAME')

def preprocess(image, mean_pixel):
    return image - mean_pixel

def undo_preprocess(image, mean_pixel):
    return image + mean_pixel

class VGG19:
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    def __init__(self):
        data = scipy.io.loadmat('imagenet-vgg-verydeep-19.mat')

        self.mean_pixel = np.array([123.68, 116.779, 103.939])

        self.weights = data['layers'][0]

    def load_weights(self):
        self.net = {}
        for i, name in enumerate(self.layers):
            kind = name[:4]
            if kind == 'conv':
                kernels = self.weights[i][0][0][2][0][0]
                bias = self.weights[i][0][0][2][0][1]

                # matconvnet: weights are [width, height, in_channels, out_channels]
                # tensorflow: weights are [height, width, in_channels, out_channels]
                kernels = np.transpose(kernels, (1, 0, 2, 3))
                bias = bias.reshape(-1)
                self.net['{}_kernel'.format(name)] = tf.Variable(kernels, \
                    name='{}_kernel'.format(name))
                self.net['{}_bias'.format(name)] = tf.Variable(bias, \
                    name='{}_bias'.format(name))

    def preprocess(self, image):
        return image-self.mean_pixel

    def undo_preprocess(self, image):
        return image+self.mean_pixel

    def feed_forward(self, input_image, scope='VGGController'):
        net = {}

        with tf.name_scope('%s_preprocess' % scope):
            current = self.preprocess(input_image)

        with tf.variable_scope(scope):
            for i, name in enumerate(self.layers):
                kind = name[:4]
                if kind == 'conv':
                    kernels = self.net['{}_kernel'.format(name)]
                    bias = self.net['{}_bias'.format(name)]
                    current = _conv_layer(current, kernels, bias)
                elif kind == 'relu':
                    current = tf.nn.relu(current)
                elif kind == 'pool':
                    current = _pool_layer(current)
                net[name] = current

        return net
