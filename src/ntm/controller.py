import tensorflow as tf
import numpy as np
from pdb import set_trace
from alexnet import AlexNet
import alexnet_OLD
from vgg19 import VGG19
from i3d import InceptionI3D

def fc_layer(input, n, activation_fn=tf.nn.relu):
    return tf.contrib.layers.fully_connected(input, n, activation_fn=activation_fn, \
        weights_regularizer=tf.nn.l2_loss)  

class DefaultController():
    def __init__(self, rnn_size, args=None):
        #  self.lstm = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
        self.lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        self.args = args

    def __call__(self, img_inp, label, vector_inp, state, scope='DefaultController'):
        img_inp = tf.reshape(img_inp, (self.args.batch_size, -1))
        controller_input = tf.concat([img_inp,label], axis=1)
        controller_input = tf.concat([controller_input]+vector_inp, axis=1)
        return self.lstm(controller_input, state)

    def zero_state(self,batch_size, dtype):
        return self.lstm.zero_state(batch_size, dtype)


class AlexNetController():

    def __init__(self, rnn_size, encoding_size, image_size=128, args=None):
        #  self.lstm = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
        self.lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        self.args = args
        # Load in the Alex net  
        self.use_pretrained = args.use_pretrained
        if self.use_pretrained:
            self.alexnet = AlexNet()
            self.alexnet.load_weights()
        else:
            self.alexnet = alexnet_OLD.AlexNet() 
        self.encoding_size = encoding_size
        self.image_size = image_size

    def __call__(self, img_inp, shifted_label, vector_inp, state, scope='AlexNetController'):
        # Q: does the img_inp need to be of 224x224?
        # Have to ensure that the input is in img form. Reshape to get the right input image size
        img_inp = tf.cast(img_inp, tf.float32)
        if self.args.dataset_type == 'omniglot':
            img_inp = tf.reshape(img_inp, [-1, self.image_size, self.image_size])
        #  img_inp = tf.stack([img_inp]*3, axis=-1)
            img_inp = tf.expand_dims(img_inp, axis=-1)
        vector_inp = tf.cast(vector_inp, tf.float32)
        net = self.alexnet.feed_forward(img_inp, architecture='encoding')
        net['flattened'] = tf.contrib.layers.flatten(net['output'])
        fc = {}
        with tf.variable_scope(scope):
            # If get casting issue make sure that the architecture is right
            fc['fc1'] = fc_layer(net['flattened'], 256)
            fc['fc2'] = fc_layer(fc['fc1'], 64)
            fc['fc3'] = fc_layer(fc['fc2'], self.encoding_size)
            fc_output = fc['fc3'] 
        lstm_input = tf.concat([fc_output,shifted_label], axis=1)
        # flatten vector_inp
        vector_inp = [vector_inp[i, :, :] for i in range(vector_inp.get_shape()[0])]
        lstm_input = tf.concat([lstm_input] + vector_inp, axis=1)
        return self.lstm(lstm_input, state)

    def zero_state(self,batch_size, dtype):
        return self.lstm.zero_state(batch_size, dtype)


class I3DController():


    def __init__(self, rnn_size, encoding_size, is_training, args=None):
        self.args = args
        self.lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        # How to take in the inputs easily
        self.encoding_size = encoding_size
        self.i3d = InceptionI3D(encoding_size, args=args, use_logits=True)
        self.is_training = is_training

    def __call__(self, img_video, shifted_label, vector_inp, state, scope='I3DController'):
        self.i3d.create_inputs_compute_graph(img_video, self.is_training)
        encoding, end_points = self.i3d.create_compute_graph(1.0)
        lstm_input = tf.concat([encoding,shifted_label], axis=1)
        # flatten vector_inp
        vector_inp = tf.convert_to_tensor(vector_inp)
        vector_inp = [vector_inp[i, :, :] for i in range(vector_inp.get_shape()[0])]
        lstm_input = tf.concat([lstm_input] + vector_inp, axis=1)
        return self.lstm(lstm_input, state)

    def zero_state(self,batch_size, dtype):
        return self.lstm.zero_state(batch_size, dtype)


class VGG19Controller():

    def __init__(self, rnn_size, encoding_size, image_size=128, args=None):
        self.lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        self.args = args
        self.vgg19 = VGG19()
        self.vgg19.load_weights()
        self.encoding_size = encoding_size
        self.image_size = image_size

    def __call__(self, img_inp, shifted_label, vector_inp, state, scope='VGG19Controller'):
        # Q: does the img_inp need to be of 224x224?
        # Have to ensure that the input is in img form. Reshape to get the right input image size
        img_inp = tf.cast(img_inp, tf.float32)
        if self.args.dataset_type == 'omniglot':
            img_inp = tf.reshape(img_inp, [-1, self.image_size, self.image_size])
        #  img_inp = tf.stack([img_inp]*3, axis=-1)
            img_inp = tf.expand_dims(img_inp, axis=-1)
        vector_inp = tf.cast(vector_inp, tf.float32)
        net = self.vgg19.feed_forward(img_inp)
        fc = {}
    
        with tf.variable_scope(scope):
            # If get casting issue make sure that the architecture is right
            net['relu5_4'] = tf.contrib.layers.flatten(net['relu5_4'])
            net['batch_norm'] = tf.layers.batch_normalization(net['relu5_4'])
            net['fc_1'] = fc_layer(net['batch_norm'], 256)
            net['fc_2'] = fc_layer(net['fc_1'], 64)
            net['output'] = fc_layer(net['fc_2'], self.encoding_size, activation_fn=None)
            fc_output = net['output'] 
        lstm_input = tf.concat([fc_output, shifted_label], axis=1)
        # flatten vector_inp
        vector_inp = [vector_inp[i, :, :] for i in range(vector_inp.get_shape()[0])]
        lstm_input = tf.concat([lstm_input] + vector_inp, axis=1)
        return self.lstm(lstm_input, state)

    def zero_state(self,batch_size, dtype):
        return self.lstm.zero_state(batch_size, dtype)
