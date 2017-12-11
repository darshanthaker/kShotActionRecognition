import numpy as np
import skvideo.io 
import imageio
import scipy.misc
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
from datetime import datetime
import sys
from pdb import set_trace

IM_SIZE = 128

def serialize_plot(lst, dirname, name):
    with open(dirname + '/' + name, 'wb') as fp:
        pickle.dump(lst, fp)

def load_file(dir_name, name):
    with open(dir_name + '/' + name, 'rb') as fp:
        lst = pickle.load(fp)
        return lst 

def gen_exp_name(args):
    return "controller:{}_dataset:{}_nclasses:{}_seqlength:{}_pretrained:{}_classdifficulty:{}_memsize:{}_memvector:{}_rnn_size:{}_img:{}"\
                .format(args.controller_type, args.dataset_type, args.n_classes, args.seq_length, args.use_pretrained, args.class_difficulty, args.memory_size, args.memory_vector_dim, args.rnn_size, args.image_height)

def mkdir(path):
    os.system("mkdir -p {}".format(path))

def eprint(*args, **kwargs):
    print(str(datetime.now().strftime('%H:%M:%S')),":", *args, file=sys.stderr, **kwargs)
    sys.stderr.flush()

def eprint2(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    sys.stderr.flush()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def sample_frames(video, sample_nframes):
    if video.shape[0] >= sample_nframes:
        all_indices = range(video.shape[0] // sample_nframes * sample_nframes)
        indices = [i % (video.shape[0] // sample_nframes) == 0 for i in all_indices]
        indices = filter(lambda x: x[1] == True, list(enumerate(indices)))
        indices = [i[0] for i in indices]
        ret_video = video[indices, :, :, :]
    else:
        ret_video = np.concatenate([np.tile(video, (sample_nframes // video.shape[0], 1, 1, 1)), \
                video[0:(sample_nframes % video.shape[0]), :, :, :]], axis=0)
    if ret_video.shape[0] != sample_nframes:
        set_trace()
    return ret_video

def video_to_frames(filename, resize=(IM_SIZE, IM_SIZE), sample_nframes=64):
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
    if sample_nframes is not None:
        video = sample_frames(video, sample_nframes)
    return video

def get_data_dir(v_type):
    return '../data/kinetics_{}/videos'.format(v_type)

def get_number_of_classes(v_type):
    assert v_type == 'train' or v_type == 'test' or v_type == 'val'
    data_dir = get_data_dir(v_type)
    labels = list()

    for label in os.listdir(data_dir):
        new_path = os.path.join(data_dir, label)
        if len(os.listdir(new_path)) == 0:
            continue
        if get_vtype_for_lab(label) != v_type:
            continue
        labels.append(label)
    return len(labels)

def get_vtype_for_lab(class_difficulty, label):
    if class_difficulty == 'all':
        train_labs_file = '../data/train_labels.npy'
        val_labs_file = '../data/test_labels.npy'
    else:
        train_labs_file = '../labels/{}_train.npy'.format(class_difficulty)
        val_labs_file = '../labels/{}_test.npy'.format(class_difficulty)
    #  train_labs_file = '../data/train_labels.npy'
    train_labs = np.load(train_labs_file)
    val_labs = np.load(val_labs_file)
    if label in train_labs:
        return 'train'
    elif label in val_labs:
        return 'val'
    else:
        return None

def get_unique_labels(v_type):
    data_dir = get_data_dir(v_type)
    unique_labels = list()
    for label in os.listdir(data_dir):
        new_path = os.path.join(data_dir, label)
        if len(os.listdir(new_path)) == 0:
            continue
        unique_labels.append(label)
    unique_labels = sorted(unique_labels)
    return unique_labels

def get_videos_lst(v_type, class_difficulty, use_subset_classes=True):
    assert v_type == 'train' or v_type == 'test' or v_type == 'val'
    data_dir = get_data_dir(v_type)
    videos_lst = list()
    labels = list()

    for label in os.listdir(data_dir):
        new_path = os.path.join(data_dir, label)
        if len(os.listdir(new_path)) == 0:
            continue
        if use_subset_classes and get_vtype_for_lab(class_difficulty, label) != v_type:
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

def find_dynamic_image(filename, resize=(IM_SIZE, IM_SIZE)):
    if not os.path.exists(filename + '.npy'): 
        raise IOError("Dynamic image for video {} does not exist".format(filename))
    im =  np.load(filename + '.npy')
    return scipy.misc.imresize(im, resize)


# For debugging
def _main():
    video_to_frames('../data/kinetics_train/videos/abseiling/13yHmtOlaUE_000751_000761.mp4')

if __name__=='__main__':
    _main()
