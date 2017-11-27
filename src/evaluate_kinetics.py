# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Loads a sample video and classifies using a trained Kinetics checkpoint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import argparse
from input_loader import InputLoader

from i3d_sonnet import *
from pdb import set_trace
from util import eprint

_IMAGE_SIZE = 224
_NUM_CLASSES = 400

_SAMPLE_PATHS = {
        'rgb': '../i3d_data/v_CricketShot_g04_c01_rgb.npy',
        'flow': '../i3d_data/v_CricketShot_g04_c01_flow.npy',
}

_CHECKPOINT_PATHS = {
        'rgb': '../i3d_data/checkpoints/rgb_scratch/model.ckpt',
        'flow': '../i3d_data/checkpoints/flow_scratch/model.ckpt',
        'rgb_imagenet': '../i3d_data/checkpoints/rgb_imagenet/model.ckpt',
        'flow_imagenet': '../i3d_data/checkpoints/flow_imagenet/model.ckpt',
}

_LABEL_MAP_PATH = '../i3d_data/label_map.txt'
_SAMPLE_VIDEO_FRAMES = 79

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('eval_type', 'joint', 'rgb, flow, or joint')
tf.flags.DEFINE_boolean('imagenet_pretrained', True, '')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=8, type=int) 
    parser.add_argument('--image_height', default=224, type=int) 
    parser.add_argument('--sample_nframes', default=79, type=int) 
    parser.add_argument('--use_subset_classes', default=False)
    parser.add_argument('--imagenet_pretrained', default=True)
    parser.add_argument('--eval_type', default='rgb') 
    args = parser.parse_args()
    eprint(args)
    evaluate_actions(args)
    #  test(args)


def define_model_rgb(args):
    # RGB input has 3 channels.
    rgb_input = tf.placeholder(
            tf.float32,
            shape=(args.batch_size, args.sample_nframes, _IMAGE_SIZE, _IMAGE_SIZE, 3))
    with tf.variable_scope('RGB'):
        rgb_model = InceptionI3d(
                _NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
        rgb_logits, _ = rgb_model(
                rgb_input, is_training=False, dropout_keep_prob=1.0)
    rgb_variable_map = {}
    for variable in tf.global_variables():
        if variable.name.split('/')[0] == 'RGB':
            rgb_variable_map[variable.name.replace(':0', '')] = variable
    rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)
    return rgb_input, rgb_model, rgb_logits, rgb_saver 

def define_model_flow(args):
    # Flow input has only 2 channels.
    flow_input = tf.placeholder(
            tf.float32,
            shape=(args.batch_size, args.sample_nframes, _IMAGE_SIZE, _IMAGE_SIZE, 2))
    with tf.variable_scope('Flow'):
        flow_model = InceptionI3d(
                _NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
        flow_logits, _ = flow_model(
                flow_input, is_training=False, dropout_keep_prob=1.0)
    flow_variable_map = {}
    for variable in tf.global_variables():
        if variable.name.split('/')[0] == 'Flow':
            flow_variable_map[variable.name.replace(':0', '')] = variable
    flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)
    return flow_input, flow_model, flow_logits, flow_saver 

def evaluate_actions(args):
    # get list of actions
    kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH)]
    data_loader = InputLoader('raw_video', 'val', args=args, im_size=args.image_height)

    labels = tf.placeholder(tf.int64, args.batch_size)
    # define the batch placeholder and the model
    if args.eval_type in ['rgb', 'joint']:
        eprint("Load RGB")
        rgb_input, rgb_model, rgb_logits, rgb_saver = define_model_rgb(args)
    if args.eval_type in ['flow', 'joint']:
        flow_input, flow_model, flow_logits, flow_saver = define_model_flow(args)

    eprint("Load Logits")
    if args.eval_type in ['rgb', 'joint']:
        model_logits = rgb_logits
    elif args.eval_type in ['flow', 'joint']:
        model_logits = flow_logits
    else:
        model_logits = rgb_logits + flow_logits
    model_predictions = tf.nn.softmax(model_logits)

    single_predictions = tf.argmax(model_predictions, axis=1)
    correct = tf.equal(single_predictions, labels)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    eprint("Start Session")
    sess = tf.Session()
    if args.imagenet_pretrained:
        if args.eval_type == 'rgb':
            rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
        elif args.eval_type == 'flow':
            flow_saver.restore(sess, _CHECKPOINT_PATHS['flow_imagenet'])
        else:
            rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
            flow_saver.restore(sess, _CHECKPOINT_PATHS['flow_imagenet'])
    else:
        if args.eval_type == 'rgb':
            rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb'])
        elif args.eval_type == 'flow':
            flow_saver.restore(sess, _CHECKPOINT_PATHS['flow'])
        else:
            rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb'])
            flow_saver.restore(sess, _CHECKPOINT_PATHS['flow'])


    #  class_indices = [57, 227, 0, 343]
    #  for action_index in class_indices:
    for action_index in range(len(kinetics_classes)):
        eprint("[{}] Loading Data".format(action_index))
        batch_videos = data_loader.sample_from_action(kinetics_classes[action_index], args.batch_size, resize=(args.image_height, args.image_height), sample_nframes=args.sample_nframes)
        batch_labels = np.array([action_index] * args.batch_size)
        feed_dict = {labels: batch_labels}
        if args.eval_type in ['rgb', 'joint']:
            feed_dict[rgb_input] = batch_videos
        if args.eval_type in ['flow', 'joint']:
            feed_dict[flow_input] = batch_videos

        eprint("[{}] Run Session".format(action_index))
        #  out_logits, out_predictions = sess.run(
                #  [model_logits, model_predictions],
                #  feed_dict=feed_dict)

        out_single_pred, out_correct, out_accuracy, out_logits, out_predictions = sess.run(
                [single_predictions, correct, accuracy, model_logits, model_predictions],
                feed_dict=feed_dict)

        #  out_predictions = out_predictions[0]
        sorted_indices = np.argsort(out_predictions, axis=-1)
        #  sorted_indices = np.argsort(out_predictions)[::-1]
        best_indices = np.argmax(out_predictions, axis=-1)
        #  set_trace()
        #  sorted_indices = sorted_indices[:, 0]
        #  correct = np.equal(sorted_indices, batch_labels)
        #  action_accuracy = np.reduce_mean(correct)
        #  set_trace()
        eprint("[{}] Action: {}. Accuracy: {:.3f}".format(action_index, kinetics_classes[action_index], out_accuracy))

        #  print('Norm of logits: %f' % np.linalg.norm(out_logits))
        #  print('\nTop classes and probabilities')
        #  for index in sorted_indices[:20]:
            #  print(out_predictions[index], out_logits[index], kinetics_classes[index])









def test(args):
    tf.logging.set_verbosity(tf.logging.INFO)
    #  eval_type = FLAGS.eval_type
    eval_type = args.eval_type
    #  imagenet_pretrained = FLAGS.imagenet_pretrained
    imagenet_pretrained = True

    if eval_type not in ['rgb', 'flow', 'joint']:
        raise ValueError('Bad `eval_type`, must be one of rgb, flow, joint')

    kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH)]

    labels = tf.placeholder(tf.int64, 1)

    if eval_type in ['rgb', 'joint']:
        # RGB input has 3 channels.
        rgb_input = tf.placeholder(
                tf.float32,
                shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))
        with tf.variable_scope('RGB'):
            rgb_model = InceptionI3d(
                    _NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
            rgb_logits, _ = rgb_model(
                    rgb_input, is_training=False, dropout_keep_prob=1.0)
        rgb_variable_map = {}
        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'RGB':
                rgb_variable_map[variable.name.replace(':0', '')] = variable
        rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

    if eval_type in ['flow', 'joint']:
        # Flow input has only 2 channels.
        flow_input = tf.placeholder(
                tf.float32,
                shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 2))
        with tf.variable_scope('Flow'):
            flow_model = InceptionI3d(
                    _NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
            flow_logits, _ = flow_model(
                    flow_input, is_training=False, dropout_keep_prob=1.0)
        flow_variable_map = {}
        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'Flow':
                flow_variable_map[variable.name.replace(':0', '')] = variable
        flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)

    if eval_type == 'rgb':
        model_logits = rgb_logits
    elif eval_type == 'flow':
        model_logits = flow_logits
    else:
        model_logits = rgb_logits + flow_logits
    model_predictions = tf.nn.softmax(model_logits)

    single_predictions = tf.argmax(model_predictions, axis=1)
    correct = tf.equal(single_predictions, labels)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    batch_labels = [227]



    with tf.Session() as sess:
        feed_dict = {}
        if eval_type in ['rgb', 'joint']:
            if imagenet_pretrained:
                rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
            else:
                rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb'])
            tf.logging.info('RGB checkpoint restored')
            rgb_sample = np.load(_SAMPLE_PATHS['rgb'])
            tf.logging.info('RGB data loaded, shape=%s', str(rgb_sample.shape))
            feed_dict[rgb_input] = rgb_sample

        if eval_type in ['flow', 'joint']:
            if imagenet_pretrained:
                flow_saver.restore(sess, _CHECKPOINT_PATHS['flow_imagenet'])
            else:
                flow_saver.restore(sess, _CHECKPOINT_PATHS['flow'])
            tf.logging.info('Flow checkpoint restored')
            flow_sample = np.load(_SAMPLE_PATHS['flow'])
            tf.logging.info('Flow data loaded, shape=%s', str(flow_sample.shape))
            feed_dict[flow_input] = flow_sample
        feed_dict[labels] = batch_labels

        #  out_logits, out_predictions = sess.run(
                #  [model_logits, model_predictions],
                #  feed_dict=feed_dict)

        out_single_pred, out_correct, out_accuracy, out_logits, out_predictions = sess.run(
                [single_predictions, correct, accuracy, model_logits, model_predictions],
                feed_dict=feed_dict)
#


        out_logits = out_logits[0]
        out_predictions = out_predictions[0]
        sorted_indices = np.argsort(out_predictions)[::-1]
        set_trace()

        print('Norm of logits: %f' % np.linalg.norm(out_logits))
        print('\nTop classes and probabilities')
        for index in sorted_indices[:20]:
            print(out_predictions[index], out_logits[index], kinetics_classes[index])


if __name__ == '__main__':
    main()
