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

import numpy as np

from keras_model_wrapper import KerasModelWrapper
from mask_rcnn import build_maskrcnn
from mask_rcnn.utils import norm_boxes, denorm_boxes, get_anchors
from data_generators.utils import mold_image, unmold_mask, compose_image_meta, resize


def mold_inputs(image, config):
    '''Takes a list of images and modifies them to the format expected
    as an input to the neural network.
    images: List of image matrices [height,width,depth]. Images can have
        different sizes.

    Returns 3 Numpy matrices:
    molded_images: [N, h, w, 3]. Images resized and normalized.
    image_metas: [N, length of meta data]. Details about each image.
    windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
        original image (padding excluded).
    '''
    window = (0, 0, image.shape[0], image.shape[1])
    scale = 1

    molded_image = mold_image(image, config.BACKBONE)

    # Build image_meta
    image_meta = compose_image_meta(
        0, image.shape, molded_image.shape, window, scale,
        np.zeros([config.NUM_CLASSES], dtype=np.int32))

    return molded_image, image_meta, window


def unmold_detections(detections, mrcnn_mask, original_image_shape,
                      image_shape, window, threshold):
    '''Reformats the detections of one image from the format of the neural
    network output to a format suitable for use in the rest of the
    application.

    detections: [N, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
    mrcnn_mask: [N, height, width, num_classes]
    original_image_shape: [H, W, C] Original image shape before resizing
    image_shape: [H, W, C] Shape of the image after resizing and padding
    window: [y1, x1, y2, x2] Pixel coordinates of box in the image where the real
            image is excluding the padding.

    Returns:
    boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
    class_ids: [N] Integer class IDs for each bounding box
    scores: [N] Float probability scores of the class_id
    masks: [height, width, num_instances] Instance masks
    '''
    # How many detections do we have?
    # Detections array is padded with zeros. Find the first class_id == 0.
    zero_ix = np.where(detections[:, 4] == 0)[0]
    N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

    # Extract boxes, class_ids, scores, and class-specific masks
    boxes = detections[:N, :4]
    class_ids = detections[:N, 4].astype(np.int32)
    scores = detections[:N, 5]
    masks = mrcnn_mask[np.arange(N), :, :, class_ids]

    # Translate normalized coordinates in the resized image to pixel
    # coordinates in the original image before resizing
    window = norm_boxes(window, image_shape)
    wy1, wx1, wy2, wx2 = window
    shift = np.array([wy1, wx1, wy1, wx1])
    wh = wy2 - wy1  # window height
    ww = wx2 - wx1  # window width
    scale = np.array([wh, ww, wh, ww])
    # Convert boxes to normalized coordinates on the window
    boxes = np.divide(boxes - shift, scale)
    # Convert boxes to pixel coordinates on the original image
    boxes = denorm_boxes(boxes, original_image_shape[:2])

    # Filter out detections with zero area. Happens in early training when
    # network weights are still random
    exclude_ix = np.where(
        (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
    if exclude_ix.shape[0] > 0:
        boxes = np.delete(boxes, exclude_ix, axis=0)
        class_ids = np.delete(class_ids, exclude_ix, axis=0)
        scores = np.delete(scores, exclude_ix, axis=0)
        masks = np.delete(masks, exclude_ix, axis=0)
        N = class_ids.shape[0]

    # Resize masks to original image size and set boundary threshold.
    full_masks = []
    for i in range(N):
        # Convert neural network mask to full size mask
        full_mask = unmold_mask(masks[i], boxes[i], original_image_shape, threshold)
        full_masks.append(full_mask)
    full_masks = np.stack(full_masks, axis=-1) \
        if full_masks else np.empty(original_image_shape[:2] + (0,))

    return boxes, class_ids, scores, full_masks


############################################################
#  MaskRCNN Class
############################################################

class MaskRCNN(KerasModelWrapper):

    def build(self):
        super(MaskRCNN, self).build()

        model, self.backbone_layer_names = build_maskrcnn(self.config)
        return model


    def predict(self, image, threshold=0.5):
        '''Runs the detection pipeline.
        images: input image

        Returns a dict that contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        '''
        super(MaskRCNN, self).predict(image, threshold)

        height, width = image.shape[:2]
        if image.shape != self.config.IMAGE_SHAPE:
            image = resize(image, output_shape=(self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH), preserve_range=True)

        # Mold inputs to format expected by the neural network
        molded_image, image_meta, window = mold_inputs(image, self.config)

        # Anchors
        anchors = get_anchors(self.config)

        # Run object detection
        # Create batch axis
        molded_images = np.expand_dims(molded_image, axis=0)
        image_metas = np.expand_dims(image_meta, axis=0)
        anchors = np.expand_dims(anchors, axis=0)

        detections, _, _, mrcnn_mask, _, _, _ = \
            self.model.predict([molded_images, image_metas, anchors], verbose=0)

        # Process detections
        rois, final_class_ids, final_scores, masks = \
            unmold_detections(detections[0], mrcnn_mask[0],
                              image.shape, molded_image.shape,
                              window, threshold)

        rate_x = width / self.config.IMAGE_WIDTH
        rate_y = height / self.config.IMAGE_HEIGHT
        final_rois = np.zeros(rois.shape)
        final_masks = np.zeros((height, width, rois.shape[0]))
        for i in range(rois.shape[0]):
            final_rois[i] = [rois[i][0] * rate_y, rois[i][1] * rate_x, rois[i][2] * rate_y, rois[i][3] * rate_x]
            final_masks[:, :, i] = resize(masks[:, :, i], (height, width))

        return {
            'rois': final_rois,
            'class_ids': final_class_ids,
            'scores': final_scores,
            'masks': final_masks,
        }
