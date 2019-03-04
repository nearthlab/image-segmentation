from collections import namedtuple

import os
import json
import numpy as np

from tqdm import tqdm
from data_generators.utils import load_image_rgb

# Copied from: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
#
# Cityscapes labels
#
#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


def label2dict(label):
    return {
        'name': label.name, 'id': label.id, 'trainId': label.trainId,
        'category': label.category, 'catId': label.categoryId, 'hasInstances': label.hasInstances,
        'ignoreInEval': label.ignoreInEval, 'color': label.color
    }


def save_labels(labels, fpath):
    l = []
    for label in labels:
        l.append(label2dict(label))

    fp = open(fpath, 'w')
    json.dump(l, fp)
    fp.close()


def load_labels(fpath):
    fp = open(fpath, 'r')
    l = json.load(fp)
    fp.close()
    labels = []
    for item in l:
        labels.append(
            Label(
                item['name'], item['id'], item['trainId'],
                item['category'], item['catId'], item['hasInstances'],
                item['ignoreInEval'], tuple(item['color']))
        )
    return labels


class KittiDataset:
    def __init__(self):
        self.image_ids = []

    def load_kitti(self, dataset_dir, subset):
        'Initialization'
        assert subset in ['train', 'val'], 'subset must be either train or val but {} is given'.format(subset)

        self.labels = load_labels(os.path.join(dataset_dir, 'label.json'))
        # color to trainId
        self.color2trainId = {label.color: label.trainId for label in self.labels}
        # trainId to name
        self.trainId2name = {label.trainId: label.name for label in self.labels}

        # number of valid trainIds + background class
        self.num_classes = max([label.trainId for label in self.labels if label.trainId >= 0 and label.trainId < 255]) + 2
        self.class_names = [self.trainId2name[i] for i in range(self.num_classes - 1)]

        self.image_dir = os.path.join(dataset_dir, subset, 'images')
        self.label_dir = os.path.join(dataset_dir, subset, 'semantic_rgb')

        assert os.path.exists(self.image_dir), 'No such directory: {}'.format(self.image_dir)
        assert os.path.exists(self.label_dir), 'No such directory: {}'.format(self.label_dir)

        self.image_files = sorted([x for x in os.listdir(self.image_dir) if x.lower().endswith('.png') or x.lower().endswith('.jpg')])
        self.label_files = sorted([x for x in os.listdir(self.label_dir) if x.lower().endswith('.png')])

        assert len(self.image_files) == len(self.label_files), \
            'image - label size mismatch! There are {} image files and {} label files'.format(len(self.image_files), len(self.label_files))

        self.num_images = len(self.image_files)
        self.image_ids = np.arange(self.num_images)

    def check_sanity(self):
        for i in tqdm(self.image_ids):
            assert self.image_files[i][:-4] == self.label_files[i][:-4],\
                'image - label filename mismatch: {} - {}'.format(self.image_files[i], self.label_files[i])
            img = load_image_rgb(os.path.join(self.image_dir, self.image_files[i]))
            msk = load_image_rgb(os.path.join(self.label_dir, self.label_files[i]))
            assert img.shape == msk.shape,\
                'img.shape: {}, msk.shape: {}'.format(img.shape, msk.shape)

    def load_image(self, image_id):
        return load_image_rgb(os.path.join(self.image_dir, self.image_files[image_id]))

    def load_mask(self, image_id):
        rgb_mask = load_image_rgb(os.path.join(self.label_dir, self.label_files[image_id]))
        mask = np.zeros((rgb_mask.shape[0], rgb_mask.shape[1], self.num_classes - 1))
        for row, col in np.ndindex(rgb_mask.shape[:2]):
            trainId = self.color2trainId[tuple(rgb_mask[row][col])]
            if trainId >= 0 and trainId != 255:
                mask[row][col][trainId] = 1.0
        return mask
