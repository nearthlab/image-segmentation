import os
import json
import numpy as np

from data_generators.utils import load_image_rgb


class ClassificationDataset:

    def __init__(self):
        self.class_names = []
        self.num_classes = 0
        self.annotations = []


    def load(self, dataset_dir, subset):
        print('Loading dataset: {} (subset: {})'.format(os.path.basename(dataset_dir), subset))
        assert os.path.isdir(dataset_dir), 'No such directory: {}'.format(dataset_dir)

        json_path = os.path.join(dataset_dir, 'class_names.json')
        assert os.path.isfile(json_path), 'No such file: {}'.format(json_path)

        with open(json_path, 'r') as fp:
            self.class_names = json.load(fp)
        self.num_classes = len(self.class_names)

        class_weights_path = os.path.join(dataset_dir, 'class_weights.json')
        try:
            fp = open(class_weights_path, 'r')
            self.class_weights = json.load(fp)
        except OSError:
            self.class_weights = [1] * self.num_classes


        assert len(self.class_weights) == self.num_classes, \
            'class_weights(={}) and num_class(={}) are inconsistent'.format(self.class_weights, self.num_classes)


        for cls in range(self.num_classes):
            image_path = os.path.join(dataset_dir, str(cls), subset)
            assert os.path.isdir(image_path), 'No such directory: {}'.format(image_path)

            image_files = sorted(os.listdir(image_path)) * self.class_weights[cls]

            for image_file in image_files:
                self.annotations.append({
                    'class': cls,
                    'path': os.path.join(image_path, image_file)
                })

            if self.class_weights[cls] == 1:
                print('class {}: num_images = {}'.format(cls, len(image_files)))
            else:
                print('class {}: num_images = {} ({} times augmented)'.format(cls, len(image_files), self.class_weights[cls]))

        self.num_images = len(self.annotations)


    def load_image(self, image_id):
        return load_image_rgb(self.annotations[image_id]['path'])


    def load_label(self, image_id):
        cls_id = self.annotations[image_id]['class']
        return np.eye(self.num_classes)[cls_id]