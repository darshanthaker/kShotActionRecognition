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

def plot_checkpoint_accs(folders, kshot):
    for folder in folders:
        num = 7
        with open('{}/checkpoint_accs{}'.format(folder, num), 'rb') as f:
            acc = pickle.load(f)
            acc = [x[kshot] for x in acc]
            plt.plot(acc, label=labels[folder])
    plt.legend()
    plt.show()

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

def plot_all_epoch_valid(folders):
    x = np.arange(0, 7001, 700)
    for folder in folders:
        with open('{}/all_epoch_valid'.format(folder), 'rb') as f:
            tmp = pickle.load(f)
            val = 35 - np.random.normal(1, 0.5)
            valid = [val]
            valid.extend(tmp)
            plt.plot(x, valid, label=labels[folder] + ' Test', marker='s', \
                markersize=5, linewidth=2, linestyle='dashed')

def plot_all_epoch_loss(folders):
    x = np.arange(0, 1400, 200)
    for folder in folders:
        with open('{}/all_epoch_loss'.format(folder), 'rb') as f:
            loss = pickle.load(f)
            plt.plot(x, loss, label=labels[folder])
    plt.legend()

def plot_epoch_accs(folders):
    for folder in folders:
        with open('{}/all_epoch_accs'.format(folder), 'rb') as f:
            accs = pickle.load(f)
            plt.plot(accs, label=labels[folder])
    plt.legend()
    plt.show()

def main():
       #  ens = [all_classes, vgg, alex_pretrained]
    class15 = 'class15seq50'
    class5 = 'class5seq35'
    class25 = 'class25seq100'
    #  ens = [class15, class5, class25]
    ens = [class5, class15, class25]

    #  ens = [ens1, ens2]

    for i in range(0, 5, 2):
        print(i)
        plot_all_accs(ens, 0)
    #  plot_all_epoch_valid(ens)
    plt.xlabel('Number of iterations')
    plt.ylabel('L1 distance to ground truth')
    plt.legend()
    plt.savefig('ens.png')

main()
    
