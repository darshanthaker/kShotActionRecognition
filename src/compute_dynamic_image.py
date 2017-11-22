import numpy as np
from util import *
from pdb import set_trace
import matplotlib.pyplot as plt

class DynamicImageGenerator(object):

    def __init__(self):
        pass

    def get_dynamic_image(self, video):
        [T, H, W, _] = video.shape
        stacked_video = np.zeros((H * W * 3, T))
        for t in range(T):
            stacked_video[:, t] = video[t, :, :, :].reshape(H * W * 3)

        for t in range(1, T + 1):
            ran = np.arange(t, T + 1, dtype=np.float32)
            alpha = np.sum((2.0 * ran - T - 1) / ran)
            stacked_video[:, t - 1] = alpha * stacked_video[:, t - 1]
        dynamic_image = np.sum(stacked_video, axis=1)
        dynamic_image = dynamic_image.reshape((H, W, 3))
        dynamic_image = dynamic_image - np.amin(dynamic_image)
        dynamic_image = 255 * dynamic_image / np.amax(dynamic_image)
        dynamic_image = dynamic_image.astype(np.uint8)
        #plt.imshow(dynamic_image)
        #plt.colorbar()
        #plt.show()
        return dynamic_image
        
    def get_dynamic_images(self, videos):
        dynamic_images = list()
        for video in videos:
            dynamic_image = get_dynamic_image(video)
            dynamic_images.append(dynamic_image)
        return dynamic_images

def main():
    #video = video_to_frames('../data/kinetics_train/videos/abseiling/13yHmtOlaUE_000751_000761.mp4')
    video = video_to_frames('~/dev/test/0347ZoDXyP0_000095_000105.mp4')
    print "Done parsing video"
    dig = DynamicImageGenerator()
    dig.get_dynamic_image(video)

main()
