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

import random
import colorsys
import numpy as np
import matplotlib.pyplot as plt

from skimage.measure import find_contours
from matplotlib import patches
from matplotlib.patches import Polygon


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def figure_to_ndarray(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)

    return buf


def find_mask_center(mask):
    sum_x, sum_y, sum_weights = 0.0, 0.0, 0.0
    for x, y in np.ndindex(mask.shape):
        sum_x += x * mask[x][y]
        sum_y += y * mask[x][y]
        sum_weights += mask[x][y]

    if sum_weights > 0:
        return sum_x / sum_weights, sum_y / sum_weights
    else:
        return None, None


def draw_instances(image, boxes, masks, class_ids, class_names=None, scores = None,
                      title="", figsize=(16, 16), show_mask=True, show_bbox=True):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    masked_image = image.astype(np.uint32).copy()

    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0],\
            'Failed expression: boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]\nboxes.shape: {}, masks.shape: {}, class_ids.shape: {}'.format(boxes.shape, masks.shape, class_ids.shape)

    fig, ax = plt.subplots(1, figsize=figsize)

    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if class_names:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[str(class_id)]
            caption = "{} {:.3f}".format(label, score) if score else label
            ax.text(x1, y1 + 8, caption,
                    color='w', size=11, backgroundcolor="none")
        else:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            caption = "{} {:.3f}".format(class_id, score) if score else str(class_id)
            ax.text(x1, y1 + 8, caption,
                    color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)

    ax.imshow(masked_image.astype(np.uint8))
    result = figure_to_ndarray(fig)
    plt.close()
    return result


def draw_segmentation(image, masks, class_names=None,
                      title="", figsize=(16, 16)):
    """
    masks: [height, width, num_classes] (num_classes excluding background)
    class_ids: [num_classes] (num_classes excluding background)
    class_names: list of class names of the dataset
    title: (optional) Figure title
    figsize: (optional) the size of the image
    """
    masked_image = image.astype(np.uint32).copy()

    # Number of instances
    N = masks.shape[-1]
    if not N:
        print("\n*** No instances to display *** \n")
    elif class_names:
        assert masks.shape[-1] == len(class_names),\
            'masks.shape: {}, class_names: {}'.format(masks.shape, class_names)

    fig, ax = plt.subplots(1, figsize=figsize)

    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    for i in range(N):
        color = colors[i]

        # Mask
        mask = masks[:, :, i]

        # Label
        if class_names:
            label = class_names[str(i)]
            x, y = find_mask_center(mask)
            if x and y:
                ax.text(y, x, label,
                    color='w', size=11, backgroundcolor="none")

        masked_image = apply_mask(masked_image, mask, colors[i])

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)

    ax.imshow(masked_image.astype(np.uint8))
    result = figure_to_ndarray(fig)
    plt.close()
    return result


def display_instances(image, boxes, masks, class_ids, class_names=None,
                      scores=None, title="", ax=None,
                      figsize=(16, 16), show_mask=True, show_bbox=True):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # If no axis is passed, create one and automatically call show()
    if ax is None:
        _, ax = plt.subplots(1, figsize=figsize)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    masked_image = draw_instances(image, boxes, masks, class_ids, class_names, scores, title, figsize, show_mask, show_bbox)
    ax.imshow(masked_image.astype(np.uint8))


def display_segmentation(image, masks, class_names=None,
                      title="", ax=None, figsize=(16, 16)):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # If no axis is passed, create one and automatically call show()
    if ax is None:
        _, ax = plt.subplots(1, figsize=figsize)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    masked_image = draw_segmentation(image, masks, class_names, title, figsize)
    ax.imshow(masked_image.astype(np.uint8))
