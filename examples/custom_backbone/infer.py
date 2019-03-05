import os
import time
import json
import argparse
import numpy as np

from skimage.io import imsave
from models import get_model_wrapper
from config import load_config
from data_generators.utils import load_image_rgb
from visual_tools import draw_instances

# Import your custom backbone
from matterport_resnet import ResNet50, ResNet101, preprocess_input

# You can add or override existing backbone model
# and corresponding preprocessing function as follows:
from classification_models import Classifiers
Classifiers._models.update({
    'resnet50': [ResNet50, preprocess_input],
    'resnet101': [ResNet101, preprocess_input],
})

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run Mask R-CNN detector.')

    parser.add_argument('-c', '--model_cfg', required=False,
                        default='MaskRCNN_coco.cfg',
                        metavar='/path/to/MaskRCNN_coco.cfg',
                        help='Path to MaskRCNN_coco.cfg file')
    parser.add_argument('-i', '--image_dir', required=False,
                        default='images',
                        metavar='/path/to/directory',
                        help='Path to a directory containing images')
    parser.add_argument('-w', '--weights', required=False,
                        default='MaskRCNN_coco.h5',
                        metavar='/path/to/MaskRCNN_coco.h5',
                        help='Path to maskrcnn weights.h5 file')
    parser.add_argument('-t', '--threshold', required=False,
                        type=float,
                        default=0.5,
                        metavar='Threshold value for inference',
                        help='Must be between 0 and 1.')
    parser.add_argument('-l', '--label', required=False,
                        type=str,
                        default='coco_class_names.json',
                        metavar='/path/to/coco_class_names.json',
                        help='Path to class json file')
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

        res = draw_instances(image=img, boxes=det['rois'],
                             masks=det['masks'], class_ids=det['class_ids'],
                             scores=det['scores'], title=image_file,
                             class_names=label)

        filename = filename[:-4] + '.png'
        imsave(os.path.join(dst_dir, filename), res)

