import numpy as np
import tensorflow as tf
from util import *
from pdb import set_trace

class DynamicImageGenerator(object):

    def __init__(self):
        self.create_compute_graph()

    def create_compute_graph(self):
        self.stacked_video = tf.placeholder(tf.float32, (None, None), name='input')
        (vec_size, T) = tf.shape(self.stacked_video)
        self.d = tf.get_variable("d", [vec_size, 1], dtype=tf.float32, \
            initializer=tf.zeros_initializer)

    def get_dynamic_image(self, video):
        [T, H, W, _] = video.shape
        stacked_video = video.reshape((H*W*3, T))
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        for epoch in range(20):
            d = self.sess.run(
                [self.d], feed_dict={self.stacked_video: stacked_video})
            set_trace()
    
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
