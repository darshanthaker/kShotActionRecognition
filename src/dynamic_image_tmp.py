import numpy as np
import tensorflow as tf
from util import *
from pdb import set_trace
import matplotlib.pyplot as plt

def eval_score(V, t, d):
    return np.dot(d, V[:, t])

def eval_loss(d, V, T, lam=0.001):
    hinge_loss = 0
    for t in range(T):
        t_score = eval_score(V, t, d)
        for q in range(t + 1, T):
            hinge_loss += max(0, 1 - eval_score(V, q, d) + t_score)
    hinge_loss = 2 * hinge_loss / float(T * (T - 1))
    regularizer = (lam / 2) * np.linalg.norm(d)**2
    return hinge_loss + regularizer

def eval_gradient(d, V, T, lam=0.001):
    grad = np.zeros(d.shape)
    x = 0
    for t in range(T):
        #grad += V[:, t] - V[: (t + 3):T]
        for q in range(t + 1, T):
            x += 1
            #grad += np.maximum(np.zeros(grad.shape), V[:, t] - V[:, q])
            grad += V[:, t] - V[:, q]
    grad = 2 * grad / float(T * (T - 1))
    grad += lam * d
    return grad

def gradient_descent(video, num_iterations=100, eta=1e-7):
    [T, H, W, _] = video.shape
    stacked_frames = video.reshape((H*W*3, T))
    V = np.zeros((H*W*3, T))
    for i in range(T):
        V[:, i] = np.sum(stacked_frames[:, 0:(i + 1)], axis=1)
    d = np.zeros(H*W*3)
    for i in range(num_iterations): 
        print "[{}] Loss = {}".format(i, eval_loss(d, V, T))
        if i % 5 == 0 and i != 0:
            resized_im = d.reshape((H, W, 3))
            plt.imshow(resized_im * 255)
            plt.colorbar()
            plt.show()
        grad = eval_gradient(d, V, T)
        d = d - (eta * grad)

def get_dynamic_image(video):
    gradient_descent(video)
    
def get_dynamic_images(videos):
    dynamic_images = list()
    for video in videos:
        dynamic_image = get_dynamic_image(video)
        dynamic_images.append(dynamic_image)
    return dynamic_images

def main():
    video = video_to_frames('../data/kinetics_train/videos/abseiling/13yHmtOlaUE_000751_000761.mp4')
    print "Done parsing video"
    get_dynamic_image(video)

main()
