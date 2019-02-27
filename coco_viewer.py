import os
import argparse
import matplotlib.pyplot as plt

from data_generators.coco.coco_dataset import CocoDataset
from visual_tools import GuiCocoViewer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Inspect COCO dataset.')
    parser.add_argument('-d', '--dataset', required=True,
                        metavar='/path/to/coco/',
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--tag', required=False,
                        default='2017',
                        metavar='<tag>',
                        help='Tag of the MS-COCO dataset (default=2017)')
    parser.add_argument('--subset', required=False,
                        default='train',
                        metavar="<subset>",
                        help='Either train or val')
    parser.add_argument('--mode', required=False,
                        default='instance',
                        metavar="<mode>",
                        help='Either instance or semantic')
    args = parser.parse_args()

    dataset_name = os.path.basename(args.dataset)
    win_titles = {'train': '{}: Training'.format(dataset_name), 'val': '{}: Validation'.format(dataset_name)}

    assert args.subset in win_titles, \
        'The argument for --subset option must be either \'train\' or \'val\' but {} is given.'.format(args.subset)

    # Load dataset
    dataset = CocoDataset()
    print('Loading subset: {} ...'.format(args.subset))
    dataset.load_coco(args.dataset, args.subset, tag=args.tag)
    dataset.prepare()

    print("Image Count: {}".format(dataset.num_images))
    print("Class Count: {}".format(dataset.num_classes))
    for i, info in enumerate(dataset.class_info):
        print("{:3}. {:50}".format(i, info['name']))

    viewer = GuiCocoViewer(
        win_titles[args.subset], dataset, args.mode)
    plt.show()
