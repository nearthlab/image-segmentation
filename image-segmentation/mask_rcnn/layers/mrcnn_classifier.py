# This file contains (modified) parts of the codes from the following repository:
# https://github.com/matterport/Mask_RCNN
#
# Mask R-CNN
#
# The MIT License (MIT)
#
# Copyright (c) 2017 Matterport, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import TimeDistributed
from keras.layers import Activation
from keras.layers import Lambda
from keras.layers import Dense
from keras.layers import Reshape

from keras.backend import int_shape
from keras.backend import squeeze

from .roi_align_layer import PyramidROIAlign

def mrcnn_classifier_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, fc_layers_size=1024):
    '''Builds the computation graph of the feature pyramid network classifier
    and regressor heads.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers
    fc_layers_size: Size of the 2 FC layers

    Returns:
        logits: [batch, num_rois, NUM_CLASSES] classifier logits (before softmax)
        probs: [batch, num_rois, NUM_CLASSES] classifier probabilities
        bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))] Deltas to apply to
                     proposal boxes
    '''
    # ROI Pooling
    # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
    x = PyramidROIAlign([pool_size, pool_size],
                        name='roi_align_classifier')([rois, image_meta] + feature_maps)
    # Two 1024 FC layers (implemented with Conv2D for consistency)
    x = TimeDistributed(Conv2D(fc_layers_size, (pool_size, pool_size), padding='valid'),
                           name='mrcnn_class_conv1')(x)
    x = TimeDistributed(BatchNormalization(), name='mrcnn_class_bn1')(x)
    x = Activation('relu')(x)
    x = TimeDistributed(Conv2D(fc_layers_size, (1, 1)),
                           name='mrcnn_class_conv2')(x)
    x = TimeDistributed(BatchNormalization(), name='mrcnn_class_bn2')(x)
    x = Activation('relu')(x)

    shared = Lambda(lambda x: squeeze(squeeze(x, 3), 2),
                       name='pool_squeeze')(x)

    # Classifier head
    mrcnn_class_logits = TimeDistributed(Dense(num_classes),
                                         name='mrcnn_class_logits')(shared)
    mrcnn_probs = TimeDistributed(Activation('softmax'),
                                  name='mrcnn_class')(mrcnn_class_logits)


    # BBox head
    # [batch, num_rois, NUM_CLASSES * (dy, dx, log(dh), log(dw))]
    x = TimeDistributed(Dense(num_classes * 4, activation='linear'),
                           name='mrcnn_bbox_fc')(shared)
    # Reshape to [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
    s = int_shape(x)
    mrcnn_bbox = Reshape((s[1], num_classes, 4), name='mrcnn_bbox')(x)

    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox