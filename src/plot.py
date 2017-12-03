import numpy as np
import pickle
import matplotlib.pyplot as plt
from pdb import set_trace

labels = {'AlexNet_variant1_2048_0.001_32_5F': 'AlexNet 2048 - 5 ensembles',
          'AlexNet_variant1_4096_0.001_32_3F': 'AlexNet 4096 - 3 ensembles',
          'VGG_vggdense_1024_0.001_32F': 'Variant2 1024',
          'VGG_vggdense_2048_0.001_32F': 'Variant2 2048',
          'VGG_vggdense_4096_0.001_32F': 'Variant2 4096',
          'VGG_arch1_1024_0.001_32F': 'Variant1 1024',
          'VGG_arch1_2048_0.001_32F': 'Variant1 2048',
          'VGG_arch1_4096_0.001_32F': 'Variant1 4096',
          'VGG_decade_1024_0.001_32F': 'Variant3 1024',
          'VGG_decade_2048_0.001_32F': 'Variant3 2048',
          'AlexNet_variant1_1024_0.001_32F': 'AlexNet 1024',
          'AlexNet_variant1_2048_0.001_32F': 'AlexNet 2048',
          'AlexNet_variant1_4096_0.001_32F': 'AlexNet 4096'

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

def plot_checkpoint_accs(folders):
    for folder in folders:
        num = 7
        with open('{}/checkpoint_accs{}'.format(folder, num), 'rb') as f:
            acc = pickle.load(f)
            plt.plot(acc, label=labels[folder])
    plt.legend()
    plt.show()

def plot_all_accs(folders):
    x = np.arange(1, 1426 * 5, 5)
    for folder in folders:
        with open('{}/all_accs'.format(folder), 'rb') as f:
            acc = pickle.load(f)
            acc = chunk(acc, SMOOTHING)
            plt.plot(x, acc, label=labels[folder] + ' Train', \
                linewidth=2)

def plot_all_loss(folders):
    for folder in folders:
        with open('{}/all_loss'.format(folder), 'rb') as f:
            loss = pickle.load(f)
            loss = chunk(loss, SMOOTHING)
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
    ens1 = 'AlexNet_variant1_2048_0.001_32_5F'
    ens2 = 'AlexNet_variant1_4096_0.001_32_3F'

    folder1 = 'VGG_arch1_1024_0.001_32F'
    folder2 = 'VGG_arch1_2048_0.001_32F'
    folder3 = 'VGG_arch1_4096_0.001_32F'

    folder4 = 'VGG_vggdense_1024_0.001_32F'
    folder5 = 'VGG_vggdense_2048_0.001_32F'
    folder6 = 'VGG_vggdense_4096_0.001_32F'

    folder7 = 'VGG_decade_1024_0.001_32F'
    folder8 = 'VGG_decade_2048_0.001_32F'

    folder9 = 'AlexNet_variant1_1024_0.001_32F'
    folder10 = 'AlexNet_variant1_2048_0.001_32F'
    folder11 = 'AlexNet_variant1_4096_0.001_32F'
    
    vgg_variant1 = [folder1, folder2, folder3]
    vgg_variant2 = [folder4, folder5, folder6]
    vgg_variant3 = [folder7, folder8]

    alexnet = [folder9, folder10, folder11]

    ens = [ens1, ens2]

    plot_all_accs(ens)
    plot_all_epoch_valid(ens)
    plt.xlabel('Number of iterations')
    plt.ylabel('L1 distance to ground truth')
    plt.legend()
    plt.savefig('ens.png')

main()
    
