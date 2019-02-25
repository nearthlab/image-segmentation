import numpy as np

# Code adopted from:
# https://github.com/matterport/Mask_RCNN/blob/4f440de25a0fbb3f7c8998c5490163699ba5aa19/mrcnn/model.py#L2798
def preprocess_input(x):
    """Expects an RGB image (or array of images) and subtracts
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return x.astype(np.float32) - np.array([123.7, 116.8, 103.9])
