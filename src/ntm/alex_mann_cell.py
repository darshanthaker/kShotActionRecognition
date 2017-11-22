import tensorflow as tf
import numpy as np
from mann_cell import MANNCell
from alexnet import AlexNet

class AlexNetController(tf.nn.rnn_cell.BasicLSTMCell):

    def __init__(self, rnn_size, encoding_size):
        tf.nn.rnn_cell.BasicLSTMCell.__init__(rnn_size)
        # Load in the Alex net  
        self.alexnet = AlexNet()
        self.encoding_size = encoding_size

    def __call__(self, inputs, state):
        encoding = self.alexnet.feed_forward(inputs, architecture='encoding')

        output = tf.contrib.nn.fully_connected(encoding, )
        return self.super().__call__(encoding, state)



class AlexMANNCell(MANNCell):

    def __init__(self, rnn_size, memory_size, memory_vector_dim, head_num, gamma=0.95, reuse=False, k_strategy='separate'):
        MANNCell.__init__(self, rnn_size, memory_size, memory_vector_dim, head_num, gamma, reuse, k_strategy)
        # replace the controller with AlexNet that passes in the encoding in to the Memory network
        self.controller = AlexNetController(rnn_size, encoding_size)

    def __call__(self, x, prev_state):
        
