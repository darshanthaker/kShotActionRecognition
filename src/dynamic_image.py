import numpy as np
import tensorflow as tf
from util import *
from pdb import set_trace
import matplotlib.pyplot as plt

class DynamicImageGenerator(object):

    def __init__(self):
        self.create_compute_graph()

    def create_compute_graph(self):
        self.stacked_video = tf.placeholder(tf.float32, (IM_SIZE**2 * 3, None), name='input')
        self.input_T = tf.shape(self.stacked_video)[1]
        # Mask to make matrix upper triangular.
        self.up_mask = tf.placeholder(tf.float32, (None, None), name='up_mask')
        # Mask to make matrix strictly upper triangular.
        self.sup_mask = tf.placeholder(tf.float32, (None, None), name='sup_mask')
        self.V = tf.placeholder(tf.float32, (IM_SIZE**2 * 3, None))
        self.d = tf.get_variable("d", [IM_SIZE**2 * 3, 1], dtype=tf.float32, \
            initializer=tf.zeros_initializer)

        self.dot_prods = tf.matmul(tf.transpose(self.d), self.V)
        self.stacked_dot_prods = tf.tile(self.dot_prods, [self.input_T, 1])
        self.up_stacked_dot_prods = tf.multiply(self.stacked_dot_prods, self.up_mask)
      
        self.W = tf.slice(self.up_stacked_dot_prods, [1, 0], \
            [self.input_T - 1, self.input_T])
        self.W = tf.concat([self.W, tf.zeros((1, self.input_T))], 0)

        self.sup_stacked_dot_prods = tf.multiply(tf.transpose(self.stacked_dot_prods), \
            self.sup_mask)

        masked_ones = tf.multiply(tf.ones((self.input_T, self.input_T)), self.sup_mask)
        self.hinge_loss_matrix = masked_ones + self.sup_stacked_dot_prods - self.W
        self.tmp_mask = tf.cast(tf.less(tf.zeros((self.input_T, \
                self.input_T)), self.hinge_loss_matrix), tf.float32)
        scale = 2.0 / tf.cast((self.input_T * (self.input_T + 1)), tf.float32)
        self.hinge_loss_matrix = scale * tf.multiply(self.hinge_loss_matrix, \
            self.tmp_mask)
        self.hinge_loss = tf.reduce_sum(self.hinge_loss_matrix)

        lam = 0.001
        self.loss = self.hinge_loss + lam * tf.nn.l2_loss(self.d)

        self.var_grad = tf.gradients(self.loss, [self.d])

        self.optimizer = tf.train.AdamOptimizer(0.001)
        #train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.minimizer = self.optimizer.minimize(self.loss)

    def get_dynamic_image(self, video):
        [T, H, W, _] = video.shape
        stacked_video = video.reshape((H*W*3, T))
        stacked_video = np.sqrt(stacked_video) # Non-linearity - Hellinger's kernel.
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        up_mask = np.triu(np.ones((T, T)))
        sup_mask = np.triu(np.ones((T, T)), k=1)

        V = np.zeros((H*W*3, T))
        for i in range(T):
            V[:, i] = (1.0 / (i + 1)) * np.sum(stacked_video[:, 0:(i + 1)], axis=1)

        for epoch in range(10000):
            loss, d, _, var_grad = self.sess.run(
                [self.loss, self.d, self.minimizer, self.var_grad], \
                feed_dict={self.stacked_video: stacked_video, \
                                     self.up_mask: up_mask, \
                                     self.sup_mask: sup_mask, \
                                     self.V: V})
            print "[{}] Loss: {}".format(epoch, loss)
            #set_trace()
            if epoch % 1000 == 0 and epoch != 0:
                resized_im = d.reshape((IM_SIZE, IM_SIZE, 3))
                resized_im = resized_im - np.amin(resized_im)
                resized_im = 255 * resized_im / np.amax(resized_im)
                plt.imshow(resized_im)
                plt.colorbar()
                plt.show()
    
    def get_dynamic_images(self, videos):
        dynamic_images = list()
        for video in videos:
            dynamic_image = get_dynamic_image(video)
            dynamic_images.append(dynamic_image)
        return dynamic_image

def main():
    video = video_to_frames('../data/kinetics_train/videos/abseiling/13yHmtOlaUE_000751_000761.mp4')
    print "Done parsing video"
    dig = DynamicImageGenerator()
    dig.get_dynamic_image(video)

main()
