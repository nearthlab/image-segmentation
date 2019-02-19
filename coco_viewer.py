import os
import argparse
import matplotlib.pyplot as plt

from data_generators.coco_dataset import CocoDataset
from data_generators.semantic.data_generator import load_image_gt
from maskrcnn.utils import extract_bboxes
from visual_tools.visualize import display_instances, display_segmentation
from visual_tools.gui_viewer import GuiViewer

class GuiCocoViewer(GuiViewer):
    def __init__(self, figurename, dataset, mode):
        super(GuiCocoViewer, self).__init__(figurename)
        assert mode in ['instance', 'semantic']
        self.mode = mode

        self.dataset = dataset
        if self.mode == 'semantic':
            self.class_names = [''] * self.dataset.num_classes
            for info in self.dataset.class_info:
                if info['id'] > 0:
                    self.class_names[info['id']] = info['name']
            self.class_names.remove('')
        self.num_images = self.dataset.num_images

        self.create_slider()
        self.create_textbox()

        self.display()

    def display(self):
        should_update = super(GuiCocoViewer, self).display()
        if should_update:
            if self.mode == 'instance':
                image = self.dataset.load_image(self.image_id)
                masks, class_ids = self.dataset.load_mask(self.image_id)

                # Compute Bounding box
                bbox = extract_bboxes(masks)

                display_instances(image, bbox, masks, class_ids,
                                  self.dataset.class_names, ax=self.ax)

                title = "ID: {}\nImage file name: {}\nThe number of objects: {}".format(
                    self.image_id,
                    os.path.basename(self.dataset.source_image_link(self.image_id)),
                    len(class_ids)
                )
            else:
                image, mask = load_image_gt(self.dataset, self.image_id, None)

                display_segmentation(image, mask, class_names=self.class_names, ax=self.ax)

                title = "ID: {}\nImage file name: {}".format(
                    self.image_id,
                    os.path.basename(self.dataset.source_image_link(self.image_id))
                )

            self.fig.suptitle(title, fontsize=20)
            self.fig.canvas.draw_idle()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Inspect COCO dataset.')
    parser.add_argument('-d', '--dataset', required=True,
                        metavar='/path/to/coco/',
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--tag', required=False,
                        default='v1',
                        metavar='<tag>',
                        help='Tag of the MS-COCO dataset (v1 or v2) (default=v1)')
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
