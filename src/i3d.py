import tensorflow
import numpy as np


def unit3d(inputs, output_channels, is_training, activation_fn=tf.nn.relu, 
        kernel_shape=(1,1,1), stride=(1,1,1), name='unit3d', use_bias=False):
    """Connects the module to inputs.

    Args:
      inputs: Inputs to the unit3d component.
      is_training: whether to use training mode for snt.BatchNorm (boolean).

    Returns:
      Outputs from the module.
    """
    with tf.variable_scope(name):
        net = tf.layers.conv3d(inputs, output_channels,
                 kernel_shape=kernel_shape,
                 stride=stride,
                 padding='SAME',
                 use_bias=False,
                 activation_fn=activation_fn)
        net = tf.contrib.layers.batch_norm(net, is_training=is_training)
    return net

def inception_module(inputs, end_point, is_training, conv_channels=[64, 96, 128, 16, 32, 32])
    with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
            branch_0 = unit3d(inputs, conv_channels[0], is_training, kernel_shape=[1, 1, 1],
              name='Conv3d_0a_1x1')

        with tf.variable_scope('Branch_1'):
            branch_1 = unit3d(inputs, conv_channels[1], is_training, kernel_shape=[1, 1, 1],
              name='Conv3d_0a_1x1')
            branch_1 = unit3d(branch_1, conv_channels[2], is_training, kernel_shape=[3, 3, 3],
              name='Conv3d_0b_3x3')

        with tf.variable_scope('Branch_2'):
            branch_2 = unit3d(inputs, conv_channels[3], is_training, kernel_shape=[1, 1, 1],
                      name='Conv3d_0a_1x1')
            branch_2 = unit3d(branch_2, conv_channels[4], is_training, kernel_shape=[3, 3, 3],
                  name='Conv3d_0b_3x3')

        with tf.variable_scope('Branch_3'):
            branch_3 = tf.nn.max_pool3d(inputs, ksize=[1, 3, 3, 3, 1],
                    strides=[1, 1, 1, 1, 1], padding='SAME',
                    name='MaxPool3d_0a_3x3')
            branch_3 = unit3d(branch_3, conv_channels[5], is_training, kernel_shape=[1, 1, 1],
                    name='Conv3d_0b_1x1')

    return tf.concat([branch_0, branch_1, branch_2, branch_3], 4)

class InceptionI3D():


    def __init__(self, args, is_logits=True, encoding_size, name='inception_i3d', spatial_squeeze=True): 
        self.args = args
        self.encoding_size = encoding_size
        self.spatial_squeeze = spatial_squeeze
        self.is_logits = is_logits

   def create_compute_graph(self, inputs, is_training, dropout_keep_prob=1.0): 
        net = inputs
        end_points = {}

        end_point = 'Conv3d_1a_7x7'
        net = unit3d(net, 64, is_training, kernel_shape=[7, 7, 7],
                     stride=[2, 2, 2], name=end_point)
        end_points[end_point] = net

        end_point = 'MaxPool3d_2a_3x3'
        net = tf.nn.max_pool3d(net, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 2, 2, 1],
                               padding='SAME', name=end_point)
        end_points[end_point] = net

        end_point = 'Conv3d_2b_1x1'
        net = unit3d(net, 64, is_training, kernel_shape=[1, 1, 1],
                     name=end_point)
        end_points[end_point] = net

        end_point = 'Conv3d_2c_3x3'
        net = Unit3D(net, 192, kernel_shape=[3, 3, 3],
                     name=end_point)

        end_point = 'MaxPool3d_3a_3x3'
        net = tf.nn.max_pool3d(net, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 2, 2, 1],
                               padding='SAME', name=end_point)
        end_points[end_point] = net

        # Start of Inception modules part
        end_point = 'Mixed_3b'
        net = inception_module(net, end_point, is_training)
        end_points[end_point] = net

        end_point = 'Mixed_3c'
        net = inception_module(net, end_point, is_training, conv_channels=[128, 128, 192, 32, 96, 64])
        end_points[end_point] = net

        end_point = 'MaxPool3d_4a_3x3'
        net = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1],
                               padding='SAME', name=end_point)
        end_points[end_point] = net

        end_point = 'Mixed_4b'
        net = inception_module(net, end_point, is_training, conv_channels=[192, 96, 208, 16, 48, 64])
        end_points[end_point] = net

        end_point = 'Mixed_4c'
        net = inception_module(net, end_point, is_training, conv_channels=[160, 112, 224, 24, 64, 64])
        end_points[end_point] = net

        end_point = 'Mixed_4d'
        net = inception_module(net, end_point, is_training, conv_channels=[128, 128, 256, 24, 64, 64])
        end_points[end_point] = net

        end_point = 'Mixed_4e'
        net = inception_module(net, end_point, is_training, conv_channels=[112, 144, 288, 32, 64, 64])
        end_points[end_point] = net

        end_point = 'Mixed_4f'
        net = inception_module(net, end_point, is_training, conv_channels=[256, 160, 320, 32, 128, 128])
        end_points[end_point] = net

        end_point = 'MaxPool3d_5a_2x2'
        net = tf.nn.max_pool3d(net, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],
                               padding='SAME', name=end_point)
        end_points[end_point] = net

        end_point = 'Mixed_5b'
        net = inception_module(net, end_point, is_training, conv_channels=[256, 160, 320, 32, 128, 128])
        end_points[end_point] = net

        end_point = 'Mixed_5c'
        net = inception_module(net, end_point, is_training, conv_channels=[384, 192, 384, 48, 128, 128])
        end_points[end_point] = net

        end_point = 'Logits'
        with tf.variable_scope(end_point):
            net = tf.nn.avg_pool3d(net, ksize=[1, 2, 7, 7, 1],
                                 strides=[1, 1, 1, 1, 1], padding='VALID')
            net = tf.contrib.layers.dropout(net, keep_prob=dropout_keep_prob, is_training=is_training)
            logits = unit3d(net, self.encoding_size, is_training
                          kernel_shape=[1, 1, 1],
                          activation_fn=None,
                          use_bias=True,
                          name='Conv3d_0c_1x1')
            if self._spatial_squeeze:
                logits = tf.squeeze(logits, [2, 3], name='SpatialSqueeze')

        averaged_logits = tf.reduce_mean(logits, axis=1)
        end_points[end_point] = averaged_logits
        if self.use_logits = True:
            return averaged_logits, end_points

        end_point = 'Predictions'
        predictions = tf.nn.softmax(averaged_logits)
        end_points[end_point] = predictions
        return predictions, end_points

    def train(self):
        pass
