import numpy as np
import pickle
import matplotlib.pyplot as plt
from pdb import set_trace

labels = {'class15seq50': 'Classes: 15, Sequence Length: 50',
        'class5seq35': 'Classes: 5, Sequence Length: 35',
        'class25seq100': 'Classes: 25, Sequence Length: 100'
        }

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

SMOOTHING = 5

def plot_all_accs(folders, kshot):
    x = np.arange(1, 40 * 25, 25)
    for folder in folders:
        print('{}/accuracy'.format(folder))
        with open('{}/accuracy'.format(folder), 'rb') as f:
            acc = pickle.load(f)
            acc = [x[kshot] for x in acc]
            plt.plot(x, acc, label=labels[folder] + ' Train', \
                linewidth=2)
    plt.legend()
    plt.show()

def plot_all_loss(folders):
    for folder in folders:
        with open('{}/loss'.format(folder), 'rb') as f:
            loss = pickle.load(f)
            plt.plot(loss, label=labels[folder])
    plt.legend()
    plt.show() 

def main():
       #  ens = [all_classes, vgg, alex_pretrained]
    class15 = 'class15seq50'
    class5 = 'class5seq35'
    class25 = 'class25seq100'
    #  ens = [class15, class5, class25]
    folders = [class5, class15, class25]

    #  ens = [ens1, ens2]

    for i in range(0, 5, 2):
        plot_all_accs(fol, 0)
    #  plot_all_epoch_valid(ens)
    plt.xlabel('Number of iterations')
    plt.ylabel('L1 distance to ground truth')
    plt.legend()
    plt.savefig('ens.png')

main()
    
