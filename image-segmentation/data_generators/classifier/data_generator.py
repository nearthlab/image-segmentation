import logging
import threading
import numpy as np

from classification_models import Classifiers
from .classification_dataset import ClassificationDataset

def data_generator(dataset: ClassificationDataset, config, batch_size, shuffle=True):
    preprocess_input = Classifiers.get_preprocessing(config.BACKBONE)

    b = 0  # batch item index
    image_index = -1
    image_ids = np.arange(dataset.num_images)
    error_count = 0

    lock = threading.Lock()
    # Keras requires a generator to run indefinitely.
    while True:
        try:
            with lock:
                # Increment index to pick next image. Shuffle if at the start of an epoch.
                image_index = (image_index + 1) % len(image_ids)
                if shuffle and image_index == 0:
                    np.random.shuffle(image_ids)

                # Get GT bounding boxes and masks for image.
                image_id = image_ids[image_index]
                image = dataset.load_image(image_id)
                label = dataset.load_label(image_id)

                # Init batch arrays
                if b == 0:
                    batch_images = np.zeros(
                        (batch_size,) + image.shape, dtype=np.float32)
                    batch_labels = np.zeros(
                        (batch_size, dataset.num_classes), dtype=np.float32)

                # Add to batch
                batch_images[b] = preprocess_input(image.astype(np.float32))
                batch_labels[b] = label.astype(np.float32)

                b += 1

                # Batch full?
                if b >= batch_size:
                    inputs = [batch_images, batch_labels]
                    outputs = []

                    yield inputs, outputs

                    # start a new batch
                    b = 0

        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(image_id))
            error_count += 1
            if error_count > 5:
                raise