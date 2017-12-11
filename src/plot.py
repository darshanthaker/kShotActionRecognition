import numpy as np
import pickle
import matplotlib.pyplot as plt
import util
import os
from pdb import set_trace

labels = {'class15seq50': 'Classes: 15, Sequence Length: 50',
        'class5seq35': 'Classes: 5, Sequence Length: 35',
        'class25seq100': 'Classes: 25, Sequence Length: 100', 
        1: '1st Instance', 
        2: '2nd Instance',
        3: '3rd Instance',
        4: '4th Instance',
        5: '5th Instance',
        6: '6th Instance',
        7: '7th Instance',
        8: '8th Instance',
        9: '9th Instance',
        10: '10th Instance'
        }

baselines = {'easy':   [0.208, 0.381, 0.4286, 0.6250, 0.4286, 0.3333, 0.6667, 0, 0], \
             'medium': [0.29, 0.25, 0.5, 0.375, 0.3, 0.5, 0.33, 0, 1, 0], \
             'hard':   [.3478, .2, .2778, .2857, .5, .4, .75, .5, .5, 1], \
             'all':    [0.48, 0.3, 0.31, 0.4, 0.42, 0.5, 0.5, 0, 0, 0]}

def chunk(x, step):
    output = list()
    count = 0
    tmp = list()
    for y in x:
        if count % step == 0 and count != 0:
            count = 0
            output.append(np.mean(np.array(tmp)))
        count += 1
        tmp.append(y)
    output.append(np.mean(np.array(tmp)))
    return np.array(output)

SMOOTHING = 2

# folder: Folder in which accuracy pickle lives.
# kshots: For number of epochs, plot how $k$-shot accuracy increase for all k in kshots.
# subfolder: Used for labeling graph.
def plot_test_accuracy_for_exp(folder, kshots, subfolder, exp_type, num_epochs=None):
    plt.clf()
    acc = util.load_file(folder, 'inter_accuracy44')
    finals = list()
    for k in kshots:
        x_acc = [a[k] for a in acc]
        chunked = chunk(x_acc, SMOOTHING)
        finals.append(chunked)
         
    step_size = 25 * SMOOTHING  # Validation accuracy calculated every STEP_SIZE epochs.
    x = np.arange(1, len(finals[0]) * step_size, step_size)
    lsubfolder = subfolder.lower()
    if num_epochs:
        comb = list(filter(lambda y: y[0] <= num_epochs, list(zip(x, acc))))
        x = [c[0] for c in comb]
        acc = [c[1] for c in comb]
        assert len(x) == len(acc)
    for (i, f) in enumerate(finals):
        plt.plot(x, f, label=labels[kshots[i] + 1], linewidth=2, marker='.', \
                 markersize=6)
    #plt.plot(x, [baselines[lsubfolder][0]] * len(acc), label='Baseline 1st Instance' ,\
    #         linewidth=2) 
    #plt.plot(x, [baselines[subfolder][1]] * len(acc), label='Baseline 2nd Instance') 
    #plt.plot(x, [baselines[subfolder][4]] * len(acc), label='Baseline 5th Instance') 
    #plt.plot(x, [baselines[lsubfolder][6]] * len(acc), label='Baseline 7th Instance', \
    #         linewidth=2) 
    plt.ylim(0, 1.1)
    plt.xlabel('Number of epochs')
    plt.ylabel('% Accurate Test Set')
    """
    if subfolder == 'vgg':
        plt.title('Pre-trained VGG19 Controller')
    else:
        plt.title('Default Controller (No encoder)')
    """
    plt.title(subfolder)
    plt.legend(loc=2)
    #plt.show()
    plt.savefig('plots/{}_{}.png'.format(exp_type, subfolder))

def plot_all_loss(folders):
    loss = util.load_file(folder, 'loss')
    for folder in folders:
        with open('{}/loss'.format(folder), 'rb') as f:
            loss = pickle.load(f)
    plt.legend()
    plt.show() 

def main():
    file_prefix = 'job_outputs/experiments/cleaned'
    exp_type = 'memory'
    subfolders = ['2x40', '64x40', '128x20', '128x40', '128x80', '128x320', '256x40']
    #subfolders = ['All', 'Easy', 'Medium', 'Hard']
    #exp_type = 'lstm'
    #subfolders = ['1', '100', '200', '300']
    #subfolders = ['hard']
    #exp_type = 'controller'
    #subfolders = ['default', 'vgg']
    #subfolders = ['no_pretrained', 'pretrained']
    folders = [os.path.join(file_prefix, exp_type, s.lower()) for s in subfolders]
    kshots = [1, 2, 5, 7, 9]

    for (subfolder, folder) in zip(subfolders, folders):
        plot_test_accuracy_for_exp(folder, kshots, subfolder, exp_type, num_epochs=None)

main()
    
