import numpy as np
import skvideo.io 
import imageio
import scipy.misc
from pdb import set_trace

def video_to_frames(filename, resize=(128, 128)):
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

# For debugging
def _main():
    video_to_frames('../data/kinetics_train/videos/abseiling/13yHmtOlaUE_000751_000761.mp4')

if __name__=='__main__':
    _main()
