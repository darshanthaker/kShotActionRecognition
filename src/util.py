import numpy as np
import skvideo.io 
import imageio
import scipy.misc
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
import sys
from pdb import set_trace

IM_SIZE = 128

def eprint(*args, **kwargs):
    print(str(datetime.now().strftime('%H:%M:%S')),":", *args, file=sys.stderr, **kwargs)

def video_to_frames(filename, resize=(IM_SIZE, IM_SIZE)):
    try:
        vid_reader = imageio.get_reader(filename,  'ffmpeg')
    except:
        eprint("Got an exception! Skipping {}".format(filename))
        return None
    t = len(vid_reader)
    assert t > 0
    (x, y, _) = vid_reader.get_data(0).shape
    if resize is not None:
        video = np.zeros([t, resize[0], resize[1], 3]) 
    else:
        video = np.zeros([t, x, y, 3])
    try:
        for i in range(len(vid_reader)):
            video[i, :, :, :] = scipy.misc.imresize(vid_reader.get_data(i), resize)
    except:
        eprint("Got an exception! Skipping {}".format(filename))
        return None
    return video

def get_data_dir(v_type):
    return '../data/kinetics_{}/videos'.format(v_type)

def get_videos_lst(v_type):
    assert v_type == 'train' or v_type == 'test' or v_type == 'val'
    data_dir = get_data_dir(v_type)
    videos_lst = list()
    labels = list()

    for label in os.listdir(data_dir):
        new_path = os.path.join(data_dir, label)
        if len(os.listdir(new_path)) == 0:
            continue
        for filename in os.listdir(new_path):
            full_file_path = os.path.join(data_dir, label, filename)
            if filename.endswith('.npy'):
                continue
            if not os.path.exists(full_file_path + '.npy'):
                # This means there was an exception loading these frames, so ignore
                # them anyways. 
                continue
            videos_lst.append(full_file_path)
            labels.append(label)

    return videos_lst, labels

def serialize(image, filename, dynamic=False):
    if dynamic:
        np.save(filename, image)
    else:
        raise NotImplementedError("Not done yet")

def one_hot_encode(x, dim):
    res = np.zeros(np.shape(x) + (dim, ), dtype=np.float32)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        res[it.multi_index][it[0]] = 1
        it.iternext()
    return res

def one_hot_decode(x):
    return np.argmax(x, axis=-1)

def find_dynamic_image(filename):
    if not os.path.exists(filename + '.npy'): 
        raise IOError("Dynamic image for video {} does not exist".format(filename))
    return np.load(filename + '.npy')


# For debugging
def _main():
    video_to_frames('../data/kinetics_train/videos/abseiling/13yHmtOlaUE_000751_000761.mp4')

if __name__=='__main__':
    _main()
