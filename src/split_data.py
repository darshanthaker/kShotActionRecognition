import sys
import numpy as np
from pdb import set_trace

data_tuples = []
with open("class_accuracies.txt", 'r') as classes:
    for line in classes:
        class_name, acc = line.split(",") 
        acc = float(acc)
        data_tuples.append((class_name, acc))

data_tuples.sort(key = lambda x: x[1])
HARD = 0.5 # gives ~100
MEDIUM = 0.75 # gives ~100
EASY = 1.0 # gives ~200

accuracies = [x for _, x in data_tuples]
hard_index = accuracies.index(HARD)
medium_index = accuracies.index(MEDIUM)

hard_class_list = np.array([data_tuples[i][0] for i in range(hard_index)])
medium_class_list = np.array([data_tuples[i][0] for i in range(hard_index, medium_index)])
easy_class_list = np.array([data_tuples[i][0] for i in range(medium_index, len(data_tuples))])

def save_split(class_list, name):
    np.random.shuffle(class_list)
    train_split = class_list[:len(class_list)//2]
    test_split = class_list[len(class_list)//2:]
    set_trace()
    np.save("{}_train.npy".format(name), train_split)
    np.save("{}_test.npy".format(name), test_split)

save_split(hard_class_list, 'hard')
save_split(medium_class_list, 'medium')
save_split(easy_class_list, 'easy')
