import numpy as np
import random
from compute_dynamic_image import DynamicImageGenerator
import util
import argparse
import os
from pdb import set_trace

class Sampler(object):


    def __init__(self, examples, sorted_labels, max_num_per_lab):
        self.examples = examples
        self.sorted_labels = sorted_labels
        self.max_num_per_lab = max_num_per_lab
        self.full_lab_examples = dict()
        for lab in self.sorted_labels:
            self.full_lab_examples[lab] = list(filter(lambda x: x[1] == lab, self.examples))
        self.clear()

    def sample(self, label):
        retval = self.labs_dict[label][self.lab_freq[label]] 
        self.lab_freq[label] += 1
        return retval

    def clear(self):
        self.labs_dict = dict()
        for lab in self.sorted_labels:
            np.random.shuffle(self.full_lab_examples[lab])
            lab_examples = self.full_lab_examples[lab][:self.max_num_per_lab]
            self.labs_dict[lab] = lab_examples
        self.lab_freq = {lab: 0 for lab in self.sorted_labels}

class InputLoader(object):

    def __init__(self, input_rep, v_type, im_size=128, use_subset_classes=True, \
            class_difficulty='easy', args=None):
        self.args = args
        self.input_rep = input_rep
        self.dig = DynamicImageGenerator()
        self.v_type = v_type
        self.im_size = im_size
        if args is not None:
            self.videos, self.labels = util.get_videos_lst(self.v_type, \
                 args.class_difficulty, \
                 use_subset_classes=args.use_subset_classes)
        else:
            self.videos, self.labels = util.get_videos_lst(self.v_type,  \
                 class_difficulty,  # just for debug purposes.
                 use_subset_classes=use_subset_classes)
        self.label_set = set(self.labels)
        self.label_lst = sorted(list(self.label_set))
        self.int_labels = [self.get_int_label(str_lab) for str_lab in self.labels]
        self.int_label_set = set(self.int_labels)
        
    def get_int_label(self, str_label):
        assert type(str_label) == str
        return int(self.label_lst.index(str_label))

    def get_str_label(self, int_label):
        assert type(int_label) == int
        return self.label_lst[int_label]

    def get_label(self, example, num_unique_classes, label_type, classes):
        label = example[1]
        if label_type == 'one_hot':
            label = classes.index(label)
            label = util.one_hot_encode(label, num_unique_classes)
        elif label_type == 'int':
            label = classes.index(label)
        return label

    def get_input(self, example):
        filename = example[0]
        if self.input_rep == 'dynamic_image':
            rep = util.find_dynamic_image(filename, resize=(self.im_size, self.im_size))
        elif self.input_rep == 'raw_video':
            rep = util.video_to_frames(filename, resize=(self.im_size, self.im_size), sample_nframes=self.args.sample_nframes)
        elif self.input_rep == 'single_frame':
            rep = util.video_to_frames(filename, resize=(self.im_size, self.im_size), sample_nframes=None)
            rep = rep[len(rep)//2, :, :, :]
        if self.args.im_normalization:
            rep = (rep/255.0 * 2.0) - 1.0
        return rep

    def fetch_serial_batch(self, batch_size):
        examples = list(zip(self.videos, self.int_labels))
        num_unique_classes = len(self.int_label_set)
        classes = sorted(list(self.int_label_set))

        np.random.shuffle(examples)
        examples = examples[:batch_size]
        batch_data = np.array([self.get_input(i) for i in examples])
        batch_labels = np.array([self.get_label(i, num_unique_classes, 'int', \
            classes) for i in examples])
        return batch_data, batch_labels

    def fetch_batch(self, num_unique_classes, batch_size, seq_length,
            augment=False,
            sampling_strategy='uniform',
            label_type='one_hot'):
        if label_type != 'one_hot' and label_type != 'int':
            raise NotImplementedError('Non one-hot encoding/int not supported yet')

        classes = random.sample(self.int_label_set, num_unique_classes)
        filtered_examples = list(filter(lambda x: x[1] in classes, \
            zip(self.videos, self.int_labels)))
        if seq_length > len(filtered_examples):
            raise ValueError("Sequence length {} too large for number of unique examples {}" \
                .format(seq_length, len(filtered_examples)))
        if sampling_strategy == 'random':
            random_indices = np.random.randint(0, len(filtered_examples), \
                batch_size * seq_length)
            examples = [filtered_examples[i] for i in random_indices]
        elif sampling_strategy == 'uniform':
            ordered_indices = np.array( \
                    [np.concatenate([\
                        [j] * int(seq_length / num_unique_classes) \
                        for j in range(num_unique_classes)])\
                     for i in range(batch_size)])
            sorted_labels = sorted(list(set([x[1] for x in filtered_examples])))
            sampler = Sampler(filtered_examples, sorted_labels, \
                int(seq_length / num_unique_classes))
            examples = list()
            for i in range(batch_size):
                sampler.clear()
                np.random.shuffle(ordered_indices[i, :])
                tmp_examples = list()
                for j in range(seq_length):
                    actual_label = sorted_labels[ordered_indices[i, j]]
                    ex = sampler.sample(actual_label)
                    tmp_examples.append(ex)
                examples += tmp_examples
            
        batch_data = list()
        batch_labels = list()
        batch_data = np.array([self.get_input(examples[i]) \
                               for i in range(batch_size * seq_length)])
        batch_labels = np.array([self.get_label(examples[i], num_unique_classes, \
                                    label_type, classes) \
                                 for i in range(batch_size * seq_length)])
        batch_data = batch_data.reshape((batch_size, seq_length) + \
            batch_data.shape[1:])
        if label_type == 'one_hot':
            batch_labels = batch_labels.reshape((batch_size, seq_length, \
                num_unique_classes))
            shifted_batch_labels = np.concatenate(
                [np.zeros(shape=[batch_size, 1, num_unique_classes]), \
                 batch_labels[:, :-1, :]], \
                axis=1)
        elif label_type =='int':
            batch_labels = batch_labels.reshape((batch_size, seq_length))
            shifted_batch_labels = None
        return batch_data, batch_labels, shifted_batch_labels

    def sample_from_action(self, action, k, resize=(128, 128), \
            sample_nframes=64):
        filtered_examples = list(filter(lambda x: x[1] == action, \
            zip(self.videos, self.labels)))
        video_names = np.array([x[0] for x in filtered_examples])
        indices = np.random.randint(len(video_names), size=k)
        sampled_names = video_names[indices]
        all_videos = list()
        for v in sampled_names:
            video = util.video_to_frames(v, resize=resize, \
                sample_nframes=sample_nframes)
            all_videos.append(video)
        all_videos = np.array(all_videos)
        if self.args.im_normalization:
            all_videos = (all_videos/255.0 * 2.0) - 1.0
        return all_videos

    def _save_all_dynamic_images(self):
        for (filename, label) in zip(self.videos, self.labels):
            if os.path.exists(filename + '.npy'):
                util.eprint("DI for {} already exists".format(filename))
                continue
            video = util.video_to_frames(filename)
            if video is None:
                continue
            dynamic_image_rep = self.dig.get_dynamic_image(video)
            util.serialize(dynamic_image_rep, filename, dynamic=True)  
            util.eprint("Serialized DI for {}".format(filename))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', default='kinetics_dynamic') # options: omniglot, kinetics_dynamic, kinetics_video, kinetics_single_frame
    parser.add_argument('--im_normalization', default=True, type=util.str2bool)
    parser.add_argument('--class_difficulty', default='all')
    parser.add_argument('--use_subset_classes', default=True, type=util.str2bool)
    args = parser.parse_args()
    input_loader = InputLoader("single_frame", "val", use_subset_classes=True, args=args)
    images, shifted_labels, y= input_loader.fetch_batch(5, 16, 35)
    set_trace()
    #input_loader._save_all_dynamic_images()

if __name__=="__main__":
    main()
