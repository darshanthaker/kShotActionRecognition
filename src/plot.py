import numpy as np
import pickle
import matplotlib.pyplot as plt
import util
import os
from pdb import set_trace

labels = {
        #  'class15seq50': 'Classes: 15, Sequence Length: 50',
        #  'class5seq35': 'Classes: 5, Sequence Length: 35',
        #  'class25seq100': 'Classes: 25, Sequence Length: 100',
        'n5seq35':"Classes: 5, Sequence Length: 35", 
        'n10seq70':"Classes: 10, Sequence Length: 70", 
        'n15seq105':"Classes: 15, Sequence Length: 105", 
        'n25seq175':"Classes: 25, Sequence Length: 175", 
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

baselines = {
            'all': [0.257, 0.2733668341708543, 0.2640163098878695, 0.23485653560042508, 0.27209302325581397, 0.2827868852459016, 0.2578947368421053, 0.23227383863080683, 0.2388663967611336, 0.2080536912751678],
            'easy': [0.281563126252505, 0.28600201409869086, 0.2792607802874743, 0.2627027027027027, 0.24191616766467067, 0.25738396624472576, 0.2382608695652174, 0.24146341463414633, 0.23773584905660378, 0.25609756097560976],
            'medium': [0.278, 0.2759315206445116, 0.25564681724845995, 0.2689434364994664, 0.24641148325358853, 0.2573426573426573, 0.2550091074681239, 0.23232323232323232, 0.2813688212927757, 0.21052631578947367],
            'hard': [0.279, 0.2598187311178248, 0.26816786079836236, 0.2505399568034557, 0.2536231884057971, 0.24859550561797752, 0.25622775800711745, 0.2422062350119904, 0.22761194029850745, 0.25748502994011974], 
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

SMOOTHING = 2

# folder: Folder in which accuracy pickle lives.
# kshots: For number of epochs, plot how $k$-shot accuracy increase for all k in kshots.
# subfolder: Used for labeling graph.
def plot_all_kshot_for_exp(folder, kshots, subfolder, exp_type, num_epochs=None):
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


def plot_single_kshot_for_exps(kshot, folders, subfolders, exp_type, num_epochs=None):
    plt.clf()
    step_size = 25
    for (subfolder, folder) in zip(subfolders, folders):
        print(kshot, subfolder)
        acc = util.load_file(folder, 'inter_accuracy44')
        x = np.arange(1, len(acc) * step_size, step_size * SMOOTHING)
        if subfolder=='VGG19' and kshot == 9:
            k_acc = [a[6] for a in acc]
        elif subfolder=='VGG19' and kshot == 7:
            k_acc = [a[5] for a in acc]
        else:
            k_acc = [a[kshot] for a in acc]
        chunked = chunk(k_acc, SMOOTHING)
        if num_epochs:
            comb = list(filter(lambda y: y[0] <= num_epochs, list(zip(x, chunked))))
            x = [c[0] for c in comb]
            chunked = [c[1] for c in comb]
            assert len(x) == len(chunked)
        if subfolder in labels:
            label = labels[subfolder]
        else:
            label = subfolder
        plt.plot(x, chunked, label=label, linewidth=2, marker='.', markersize=6)
    if exp_type == 'difficulty':
        plt.plot(x, [baselines[subfolder][kshot]] * len(chunked), label='Baseline')
    else:
        x = np.arange(1, num_epochs, step_size * SMOOTHING)
        #  set_trace()
        plt.plot(x, [baselines['all'][kshot]] * len(chunked), label='Baseline')
    plt.ylim(0.0, 0.8)
    plt.xlabel('Number of epochs')
    plt.ylabel('% Accurate Test Set')
    plt.title("{}-Shot Accuracy".format(kshot))
    plt.legend(loc=2)
    #plt.show()
    plt.savefig('plots/{}_{}.png'.format(exp_type, kshot))

def plot_all_loss(folders):
    loss = util.load_file(folder, 'loss')
    for folder in folders:
        with open('{}/loss'.format(folder), 'rb') as f:
            loss = pickle.load(f)
    plt.legend()
    plt.show() 

def main():
    file_prefix = 'job_outputs/experiments/cleaned'
    #  exp_type = 'memory'
    #  exp_type = 'ablation'
    exp_type = 'controllers'
    #  exp_type = 'classes'
    #  subfolders = ['2x40', '64x40', '128x20', '128x40', '128x80', '128x320', '256x40']
    subfolders = ['AlexNet No Pre-Training', 'AlexNet With Pre-training', 'VGG19', 'No Encoder']
    #  subfolders = []
    #  subfolders = ['All', 'Easy', 'Medium', 'Hard']
    #  subfolders = ['n5seq35', 'n10seq70', 'n15seq105', 'n25seq175']
    #exp_type = 'lstm'
    #subfolders = ['1', '100', '200', '300']
    #subfolders = ['hard']
    #exp_type = 'controller'
    #subfolders = ['default', 'vgg']
    #subfolders = ['no_pretrained', 'pretrained']
    folders = [os.path.join(file_prefix, exp_type, s.lower()).replace(" ", "_") for s in subfolders]
    kshots = [1, 2, 5, 7, 9]

    for i in kshots:
        plot_single_kshot_for_exps(i, folders, subfolders, exp_type, num_epochs=4000)

    #  for (subfolder, folder) in zip(subfolders, folders):
        #  plot_test_accuracy_for_exp(folder, kshots, subfolder, exp_type, num_epochs=None)

main()
    
