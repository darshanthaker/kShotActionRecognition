import tensorflow as tf
import numpy as np
from alexnet import AlexNet

def fc_layer(input, n, activation_fn=tf.nn.relu):
    return tf.contrib.layers.fully_connected(input, n, activation_fn=activation_fn, \
        weights_regularizer=tf.nn.l2_loss)  

class DefaultController():
    def __init__(self, rnn_size):
        self.lstm = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)

    def __call__(self, img_inp, label, vector_inp, state, scope='AlexNetController'):
    #  def __call__(self, img_inp, vector_inp, state, scope='DefaultController'):
        controller_input = tf.concat([img_inp,label], axis=1)
        controller_input = tf.concat([controller_input]+vector_inp, axis=1)
        return self.lstm(controller_input, state)

    def zero_state(self,batch_size, dtype):
        return self.lstm.zero_state(batch_size, dtype)


class AlexNetController():

    def __init__(self, rnn_size, encoding_size, image_size=20):
        self.lstm = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
        # Load in the Alex net  
        self.alexnet = AlexNet()
        self.encoding_size = encoding_size
        self.image_size = image_size

    def __call__(self, img_inp, shifted_label, vector_inp, state, scope='AlexNetController'):
        # Q: does the img_inp need to be of 224x224?
        # Have to ensure that the input is in img form. Reshape to get the right input image size
        img_inp = tf.cast(img_inp, tf.float32)
        img_inp = tf.reshape(img_inp, [-1, self.image_size, self.image_size])
        #  img_inp = tf.stack([img_inp]*3, axis=-1)
        img_inp = tf.expand_dims(img_inp, axis=-1)
        vector_inp = tf.cast(vector_inp, tf.float32)
        net = self.alexnet.feed_forward(img_inp, architecture='encoding')
        fc = {}
        with tf.variable_scope(scope):
            # If get casting issue make sure that the architecture is right
            fc['fc1'] = fc_layer(net['output'], 256)
            fc['fc2'] = fc_layer(fc['fc1'], 64)
            fc['fc2'] = fc_layer(fc['fc1'], self.encoding_size)
            fc_output = fc['fc2'] 
        lstm_input = tf.concat([fc_output,shifted_label], axis=1)
        # flatten vector_inp
        vector_inp = [vector_inp[i, :, :] for i in range(vector_inp.get_shape()[0])]
        lstm_input = tf.concat([lstm_input] + vector_inp, axis=1)
        return self.lstm(lstm_input, state)

    def zero_state(self,batch_size, dtype):
        return self.lstm.zero_state(batch_size, dtype)




#  class AlexMANNCell(MANNCell):
#
    #  def __init__(self, rnn_size, memory_size, memory_vector_dim, head_num, gamma=0.95, reuse=False, k_strategy='separate'):
        #  MANNCell.__init__(self, rnn_size, memory_size, memory_vector_dim, head_num, gamma, reuse, k_strategy)
        #  # replace the controller with AlexNet that passes in the encoding in to the Memory network
        #  self.controller = AlexNetController(rnn_size, encoding_size)
#
    #  def __call__(self, x, prev_state):
        #
