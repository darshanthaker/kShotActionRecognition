import numpy as np
import random
from compute_dynamic_image import DynamicImageGenerator
import util
import os
from pdb import set_trace

class InputLoader(object):

    def __init__(self, input_rep, v_type):
        self.input_rep = input_rep
        self.dig = DynamicImageGenerator()
        self.v_type = v_type
        self.videos, self.labels = util.get_videos_lst(self.v_type)
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
        return label

    def get_input(self, example):
        filename = example[0]
        if self.input_rep == 'dynamic_image':
            rep = util.find_dynamic_image(filename)
        return rep

    def fetch_batch(self, num_unique_classes, batch_size, seq_length,
            augment=False,
            sampling_strategy='random',
            label_type='one_hot'):
        if label_type != 'one_hot':
            raise NotImplementedError('Non one-hot encoding not supported yet')

        if sampling_strategy == 'random':
            classes = random.sample(self.int_label_set, num_unique_classes)
            filtered_examples = list(filter(lambda x: x[1] in classes, \
                zip(self.videos, self.int_labels)))
            if seq_length > len(filtered_examples):
                raise ValueError("Sequence length {} too large for number of unique examples {}" \
                    .format(seq_length, len(filtered_examples)))
        batch_data = list()
        batch_labels = list()
        random_indices = np.random.randint(0, len(filtered_examples), \
            batch_size * seq_length)
        examples = [filtered_examples[i] for i in random_indices]

        batch_data = np.array([self.get_input(examples[i]) \
                               for i in range(batch_size * seq_length)])
        batch_labels = np.array([self.get_label(examples[i], num_unique_classes, \
                                    label_type, classes) \
                                 for i in range(batch_size * seq_length)])
        batch_data = batch_data.reshape((batch_size, seq_length) + \
            batch_data.shape[1:])
        batch_labels = batch_labels.reshape((batch_size, seq_length, \
            num_unique_classes))
        shifted_batch_labels = np.concatenate(
            [np.zeros(shape=[batch_size, 1, num_unique_classes]), \
             batch_labels[:, :-1, :]], \
            axis=1)
        return batch_data, batch_labels, shifted_batch_labels

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
    input_loader = InputLoader("dynamic_image", "train")
    #input_loader.fetch_batch(2, 4, 4)
    input_loader._save_all_dynamic_images()

if __name__=="__main__":
    main()
