from __future__ import print_function, absolute_import, division
from collections import namedtuple

import os
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

labels = [
            #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
            Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
            Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
            Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
            Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
            Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
            Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
            Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
            Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
            Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
            Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
            Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
            Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
            Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
            Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
            Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
            Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
            Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
            Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
            Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
            Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
            Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
            Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
            Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
            Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
            Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
            Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
            Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
            Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
            Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
            Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
            Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
            Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
            Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
            Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
            Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
        ]

# color to trainId
color2trainId = {label.color: label.trainId for label in labels}
# trainId to name
trainId2name = {label.trainId: label.name for label in labels}

class KittiDataset:
    def __init__(self):
        self.image_ids = []

        # number of valid trainIds + background class
        self.num_classes = len([label for label in labels if label.trainId >= 0 and label.trainId != 255]) + 1

    def load_kitti(self, dataset_dir, subset):
        'Initialization'
        assert subset in ['train', 'val'], 'subset must be either train or val but {} is given'.format(subset)

        self.image_dir = os.path.join(dataset_dir, subset, 'images')
        self.label_dir = os.path.join(dataset_dir, subset, 'semantic_rgb')

        assert os.path.exists(self.image_dir), 'No such directory: {}'.format(self.image_dir)
        assert os.path.exists(self.label_dir), 'No such directory: {}'.format(self.label_dir)

        self.image_files = sorted([x for x in os.listdir(self.image_dir) if x.endswith('.png')])
        self.label_files = sorted([x for x in os.listdir(self.label_dir) if x.endswith('.png')])

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
            trainId = color2trainId[tuple(rgb_mask[row][col])]
            if trainId >= 0 and trainId != 255:
                mask[row][col][trainId] = 1.0
        return mask
