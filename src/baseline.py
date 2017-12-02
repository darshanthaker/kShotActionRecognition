import numpy as np
from sklearn.cluster import KMeans
from util import *
from input_loader import InputLoader

def main():
    val_input_loader = InputLoader('dynamic_image', 'val')
    num_unique_classes = 10
    batch_size = 256
    num_batches = 1
    seq_length = 35

    print("Number of unique classes: {}, Batch size: {}, Sequence length: {}".format( \
        num_unique_classes, batch_size, seq_length))

    all_labs = list()
    all_pred_labs = list()
    for i in range(batch_size):
        batch_data, batch_labels, _ = val_input_loader.fetch_batch(num_unique_classes, \
            batch_size, seq_length, label_type='int')

        batch_data = batch_data.reshape((batch_size, seq_length, -1))
        feat_vecs = batch_data[i, :, :]
        labs = batch_labels[i, :]
        kmeans = KMeans(n_clusters=num_unique_classes).fit(feat_vecs)
        kmeans_labs = kmeans.labels_
        vote_map = dict()
        for (f, l) in zip(kmeans_labs, labs):
            if f not in vote_map:
                vote_map[f] = [l]
            else:
                vote_map[f].append(l)
        label_map = dict()
        for (f, votes) in vote_map.items():
            tmp_map = dict()
            for vote in votes:
                if vote not in tmp_map:
                    tmp_map[vote] = 1
                else:
                    tmp_map[vote] += 1
            label_map[f] = min(tmp_map.items(), key=lambda x: x[1])[0]
        pred_labs = [label_map[i] for i in kmeans_labs]
        #all_labs += list(labs)
        #all_pred_labs += list(pred_labs)
        all_labs = list(labs)
        all_pred_labs = pred_labs

    correct = [0] * seq_length
    total = [0] * seq_length
    class_count = {}
    for j in range(len(all_labs)):
        if all_labs[j] not in class_count:
            class_count[all_labs[j]] = 0
        class_count[all_labs[j]] += 1
        total[class_count[all_labs[j]]] += 1
        if all_labs[j] == all_pred_labs[j]:
            correct[class_count[all_labs[j]]] += 1
    acc = [float(correct[i]) / total[i] if total[i] > 0. else 0. for i in range(1, 11)]
    print("Test Result\n1st\t2nd\t3rd\t4th\t5th\t6th\t7th\t8th\t9th\t10th")
    for a in acc:
        print('%.4f' % a, end='\t')
    print("")



if __name__=="__main__":
    main()
