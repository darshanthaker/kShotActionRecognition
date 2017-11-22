import numpy as np
import random
from compute_dynamic_image import DynamicImageGenerator
from util import *

class InputLoader(object):

    def __init__(self, input_rep, v_type):
        self.input_rep = input_rep
        self.dig = DynamicImageGenerator()
        self.v_type = v_type
        self.videos, self.labels = get_videos_lst(self.v_type)
        self.label_set = set(self.labels)

    def get_input_and_label(self, example, label_encoding='int'):
        filename = example[0]
        # TODO(dbthaker): Do some different label encodings.
        label = example[1]
        if self.input_rep == 'dynamic_image':
            video = video_to_frames(filename)
            dynamic_image_rep = self.dig.get_dynamic_image(video)
            return dynamic_image_rep, label

    def get_next_batch(self, batch_size, 
            sampling_strategy='random', \
            num_unique_classes=10):
        if sampling_strategy == 'random':
            classes = random.sample(self.label_set, num_unique_classes)
            examples = list(filter(lambda x: x[1] in classes, \
                zip(self.videos, self.labels)))
            if batch_size > len(examples):
                print("Batch size too large for number of unique classes.")
        batch_data = list()
        batch_labels = list()
        np.random.shuffle(examples)

        for i in range(batch_size):
            data, label = get_input_and_label(examples[i])
            batch_data.append(data)
            batch_labels.append(label)
        return batch_data, batch_labels
