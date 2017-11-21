import numpy as np
import skvideo.io 
import imageio
import scipy.misc
import matplotlib.pyplot as plt
import pandas as pd
import os
from pdb import set_trace

IM_SIZE = 128

def video_to_frames(filename, resize=(IM_SIZE, IM_SIZE)):
    vid_reader = imageio.get_reader(filename,  'ffmpeg')
    t = len(vid_reader)
    assert t > 0
    (x, y, _) = vid_reader.get_data(0).shape
    if resize is not None:
        video = np.zeros([t, resize[0], resize[1], 3]) 
    else:
        video = np.zeros([t, x, y, 3])
    for i in range(len(vid_reader)):
        video[i, :, :, :] = scipy.misc.imresize(vid_reader.get_data(i), resize)
    return video

def get_data_dir(v_type):
    return '../data/kinetics_{}/videos'.format(v_type)

def get_videos_lst(v_type):
    assert v_type == 'train' or v_type == 'test' or v_type == 'val'
    data_dir = get_data_dir(v_type)
    videos_lst = list()
    labels = list()

    for label in os.listdir(data_dir):
        if len(os.listdir(label)) == 0:
            continue
        for filename in os.listdir(label):
            videos_lst.append(filename)
            labels.append(label)

    return videos_lst, labels

# For debugging
def _main():
    video_to_frames('../data/kinetics_train/videos/abseiling/13yHmtOlaUE_000751_000761.mp4')

if __name__=='__main__':
    _main()
