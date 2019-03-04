import os
import time
import json
import argparse
import numpy as np

from skimage.io import imsave
from models import get_model_wrapper
from config import load_config
from data_generators.utils import load_image_rgb
from visual_tools import draw_segmentation, draw_instances

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run inference and visualize the result.')

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
    parser.add_argument('-t', '--threshold', required=False,
                        type=float,
                        default=0.5,
                        metavar='Threshold value for inference',
                        help='Must be between 0 and 1.')
    parser.add_argument('-l', '--label', required=False,
                        type=str,
                        default='label.json',
                        metavar='/path/to/label.json',
                        help='Path to label json file')
    args = parser.parse_args()

    assert args.threshold >= 0.0 and args.threshold < 1.0, \
        'Invalid threshold value {} given'.format(args.threshold)

    if args.label is not None:
        assert os.path.exists(args.label)
        fp = open(args.label, 'r')
        label = json.load(fp)
        fp.close()
    else:
        label = None

    model = get_model_wrapper(load_config(args.model_cfg))
    if args.weights:
        model.load_weights(args.weights)

    image_files = sorted([os.path.join(args.image_dir, x)
                               for x in os.listdir(args.image_dir)
                               if x.lower().endswith('.jpg') or x.lower().endswith('.png') or x.lower().endswith('.bmp')
                               ])

    dst_dir = os.path.join(args.image_dir, 'results')
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    for image_file in image_files:
        filename = os.path.basename(image_file)
        print('Processing {}...'.format(filename))
        img = load_image_rgb(image_file)
        t = time.time()
        det = model.predict(img.astype(np.float32), args.threshold)
        time_taken = time.time() - t
        print("\ttime taken: {}".format(time_taken))

        if model.config.MODEL == 'maskrcnn':
            res = draw_instances(image=img, boxes=det['rois'],
                              masks=det['masks'], class_ids=det['class_ids'],
                              scores=det['scores'], title=image_file,
                              class_names=label)
        else:
            res = draw_segmentation(image=img, masks=det, class_names=label)

        imsave(os.path.join(dst_dir, filename), res)

