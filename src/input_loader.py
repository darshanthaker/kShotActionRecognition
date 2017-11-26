import numpy as np
import random
from compute_dynamic_image import DynamicImageGenerator
import util
import os
from pdb import set_trace

class InputLoader(object):

    def __init__(self, input_rep, v_type, im_size=128, args=None):
        self.args = args
        self.input_rep = input_rep
        self.dig = DynamicImageGenerator()
        self.v_type = v_type
        self.im_size = im_size
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
        elif label_type == 'int':
            label = classes.index(label)
        return label

    def get_input(self, example):
        filename = example[0]
        if self.input_rep == 'dynamic_image':
            rep = util.find_dynamic_image(filename)
        elif self.input_rep == 'raw_video':
            rep = util.video_to_frames(filename, resize=(self.im_size, self.im_size), sample_nframes=args.sample_nframes)
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
        if label_type != 'one_hot':
            raise NotImplementedError('Non one-hot encoding not supported yet')

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
            for i in range(batch_size):
                np.random.shuffle(ordered_indices[i, :])
            # TODO(dbthaker): Finish this shit.
            #for i in np.nditer(ordered_indices):
            #    examples.append(
            #examples = [filtered_examples[i] for i in np.nditer(ordered_indices)]
            
        batch_data = list()
        batch_labels = list()
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
    input_loader = InputLoader("dynamic_image", "val")
    #input_loader.fetch_batch(2, 4, 4)
    input_loader._save_all_dynamic_images()

if __name__=="__main__":
    main()
