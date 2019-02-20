import logging
import numpy as np

from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage.morphology import distance_transform_edt
from skimage.filters import sobel

from data_generators.utils import resize_image, resize_mask
from classification_models import Classifiers


def load_image_gt(dataset, image_id, image_size):
    """Load and return ground truth data for an image (image, mask, bounding boxes).

    Returns:
    image: [height, width, 3]
    mask: [height, width, NUM_CLASSES].
    NUM_CLASSES doesn't include background
    """
    # Load image and mask
    image = dataset.load_image(image_id)
    masks, class_ids = dataset.load_mask(image_id)

    # Resize image and mask
    if image_size:
        image, window, scale, padding, crop = resize_image(image, image_size)
        masks = resize_mask(masks, scale, padding, crop)

    # erase intersection to prevent masks of different instances in the same class from overlapping
    for c1 in range(masks.shape[-1] - 1):
        for c2 in range(c1 + 1, masks.shape[-1]):
            intersection = binary_dilation(masks[:, :, c1]) * binary_dilation(masks[:, :, c2])
            not_intersection = np.logical_not(binary_dilation(intersection))
            masks[:, :, c1] = np.logical_and(masks[:, :, c1], not_intersection)
            masks[:, :, c2] = np.logical_and(masks[:, :, c2], not_intersection)

    # # erode masks for preventing overlapping masks of different instances in the same class
    # for c in range(masks.shape[-1]):
    #     masks[:, :, c] = binary_erosion(masks[:, :, c], iterations=2).astype(masks.dtype)

    return image, merge_masks(masks, class_ids, dataset.num_classes - 1)


def get_weight_mask(mask, eps=0.5):
    weight_mask = np.zeros(mask.shape)

    for c in range(mask.shape[-1]):
        sobel_edge = sobel(mask[:, :, c])
        sobel_edge[sobel_edge > 0] = 1.0
        sobel_edge = 1.0 - sobel_edge
        edt_map = distance_transform_edt(sobel_edge)
        weight_mask[:, :, c] = 1.0 - eps * np.interp(edt_map, (edt_map.min(), edt_map.max()), (0.0, 1.0))

    return np.mean(weight_mask, axis=-1)

# num_classes: exclude background
def merge_masks(masks, class_ids, num_classes):
    image_dim = masks.shape[0:2]
    merged_masks = [np.zeros(shape=image_dim)] * num_classes

    for i in range(masks.shape[-1]):
        merged_masks[class_ids[i] - 1] = np.maximum(merged_masks[class_ids[i] - 1], masks[:, :, i])

    merged_mask = np.zeros(shape=(*image_dim, num_classes))
    for i in range(num_classes):
        merged_mask[:, :, i] = merged_masks[i]

    return merged_mask


def data_generator(dataset, config, shuffle=True, batch_size=1):
    """A generator that returns images and corresponding target class ids,
    bounding box deltas, and masks.

    dataset: The Dataset object to pick data from
    config: The model config object
    shuffle: If True, shuffles the samples before every epoch
    batch_size: How many images to return in each call

    Returns a Python generator. Upon calling next() on it, the
    generator returns two lists, inputs and outputs. The contents
    of the lists differs depending on the received arguments:
    inputs list:
    - images: [batch, H, W, C]
    - gt_mask: [batch, height, width, NUM_CLASSES]. The height and width
                are those of the image and NUM_CLASSES doesn't include background

    outputs list: empty
    """
    b = 0  # batch item index
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0

    preprocess_input = Classifiers.get_preprocessing(config.BACKBONE)

    # Keras requires a generator to run indefinitely.
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            # Get GT bounding boxes and masks for image.
            image_id = image_ids[image_index]
            image, gt_mask = load_image_gt(dataset, image_id, config.IMAGE_SIZE)
            weight_mask = get_weight_mask(gt_mask)

            # Init batch arrays
            if b == 0:
                batch_images = np.zeros(
                    (batch_size,) + image.shape, dtype=np.float32)
                batch_gt_mask = np.zeros(
                    (batch_size, gt_mask.shape[0], gt_mask.shape[1],
                     config.NUM_CLASSES - 1), dtype=gt_mask.dtype)
                batch_weight_mask = np.zeros(
                    (batch_size, weight_mask.shape[0], weight_mask.shape[1]),
                    dtype=gt_mask.dtype)

            # Add to batch
            batch_images[b] = preprocess_input(image.astype(np.float32))
            batch_gt_mask[b, :, :, :] = gt_mask
            batch_weight_mask[b, :, :] = weight_mask

            b += 1

            # Batch full?
            if b >= batch_size:
                inputs = [batch_images, batch_gt_mask, batch_weight_mask]
                outputs = []

                yield inputs, outputs

                # start a new batch
                b = 0

        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(
                dataset.image_info[image_id]))
            error_count += 1
            if error_count > 5:
                raise
