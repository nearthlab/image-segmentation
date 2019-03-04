import os
import argparse
import matplotlib.pyplot as plt

from data_generators.kitti import KittiDataset
from visual_tools import GuiKittiViewer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Inspect COCO dataset.')
    parser.add_argument('-d', '--dataset', required=True,
                        metavar='/path/to/KITTI/',
                        help='Directory of the KITTI dataset')
    parser.add_argument('--subset', required=False,
                        default='train',
                        metavar="<subset>",
                        help='Either train or val')
    parser.add_argument('--check_sanity', required=False,
                        default=False,
                        type=bool,
                        metavar='<check-sanity>',
                        help='Whether to check sanity of the dataset or not')
    args = parser.parse_args()

    dataset_name = os.path.basename(args.dataset)
    win_titles = {'train': '{}: Training'.format(dataset_name), 'val': '{}: Validation'.format(dataset_name)}

    assert args.subset in win_titles, \
        'The argument for --subset option must be either \'train\' or \'val\' but {} is given.'.format(args.subset)

    # Load dataset
    dataset = KittiDataset()
    print('Loading subset: {} ...'.format(args.subset))
    dataset.load_kitti(args.dataset, args.subset)
    if args.check_sanity:
        print('Checking sanity of the dataset...')
        dataset.check_sanity()

    print("Image Count: {}".format(dataset.num_images))
    print("Class Count: {}".format(dataset.num_classes))
    for i, name in enumerate(dataset.class_names):
        print("{:3}. {:50}".format(i, name))

    viewer = GuiKittiViewer(
        win_titles[args.subset], dataset)
    plt.show()
