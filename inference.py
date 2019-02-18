import os
import time
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

from models import get_model_wrapper
from config import load_config
from data_generators.utils import load_image_rgb

if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run Mask R-CNN detector.')

    parser.add_argument('-c', '--model_cfg', required=True,
                        metavar='/path/to/model.cfg',
                        help='Path to model.cfg file')
    parser.add_argument('-i', '--image_dir', required=True,
                        metavar='/path/to/directory',
                        help='Path to a directory containing images')
    parser.add_argument('-w', '--weights', required=False,
                        default=None,
                        metavar='/path/to/weights.h5',
                        help='Path to maskrcnn weights.h5 file')
    parser.add_argument('-l', '--limit', required=False,
                        default=5, type=int,
                        metavar='Maximum number of images to display',
                        help='Images to use for display results (default=5)')
    args = parser.parse_args()

    labels = {0: 'background', 1: 'blade'}

    # Model Configurations
    model_config = load_config(args.model_cfg)

    # Create model
    print('Building model...')
    model = get_model_wrapper(model_config)
    if args.weights is not None:
        print('Loading {}...'.format(args.weights))
        model.load_weights(args.weights)
    else:
        print('Weights path is not specified. Will initialize network with random weights')

    image_dir = args.image_dir
    assert os.path.isdir(image_dir), 'No such directory: {}'.format(image_dir)
    # Get the list of image files
    image_files = [os.path.join(image_dir, x)
                   for x in os.listdir(image_dir)
                   if x.lower().endswith('.jpg') or x.lower().endswith('.png') or x.lower().endswith('.bmp')]

    if len(image_files) > args.limit:
        print('{} images found. Will display results for only {} randomly chosen images'.format(len(image_files), args.limit))
        image_files = random.sample(set(image_files), args.limit)

    if model.config.MODEL == 'maskrcnn':
        from tools.visualize import display_instances as display
    else:
        from tools.visualize import display_segmentation as display

    for image_file in image_files:
        print('Processing {}...'.format(image_file))
        img = load_image_rgb(image_file)

        t = time.time()
        det = model.predict(img.astype(np.float32))
        time_taken = time.time() - t
        print("\ttime taken: {}".format(time_taken))

        if model.config.MODEL == 'maskrcnn':
            display(image=img, boxes=det['rois'], masks=det['masks'], class_ids=det['class_ids'], scores=det['scores'], class_names=labels, title=image_file)
        else:
            display(image=img, masks=det)

    plt.show()

