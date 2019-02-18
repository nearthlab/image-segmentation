import os
import argparse

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.widgets import TextBox

from data_generators.coco_dataset import CocoDataset
from data_generators.semantic.data_generator import load_image_gt
from maskrcnn.utils import extract_bboxes
from tools.visualize import display_instances, display_segmentation

class GuiCocoViewer:
    def __init__(self, dataset, figurename, mode):
        assert mode in ['instance', 'semantic']
        self.mode = mode
        self.fig, self.ax = plt.subplots(1, figsize=(16, 16), num=figurename)
        self.image_id = 0
        self.prev_image_id = None
        self.dataset = dataset
        if self.mode == 'semantic':
            self.class_names = [''] * self.dataset.num_classes
            for info in self.dataset.class_info:
                if info['id'] > 0:
                    self.class_names[info['id']] = info['name']
            self.class_names.remove('')
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)

        self.fig.canvas.mpl_connect('key_press_event', self.keypress_callback)

        self.axcolor = 'lightgoldenrodyellow'
        ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor=self.axcolor)
        self.slider = Slider(ax_slider, 'Id', 0, self.dataset.num_images - 1, valinit=0)
        self.slider.on_changed(self.slider_callback)

        axbox = plt.axes([0.2, 0.05, 0.65, 0.03])
        self.text_box = TextBox(axbox, '', initial=str(self.image_id))
        self.text_box.on_submit(self.text_box_callback)
        self.deactivate_text_box()

        self.display_image_and_label()
        
    def display_image_and_label(self):
        if self.prev_image_id != self.image_id:
            self.ax.clear()

            print('image id: {}'.format(self.image_id))
            if self.mode == 'instance':
                self.prev_image_id = self.image_id
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

                self.fig.suptitle(title, fontsize=20)
                self.fig.canvas.draw_idle()
            else:
                self.prev_image_id = self.image_id
                image, mask = load_image_gt(self.dataset, self.image_id, None)

                display_segmentation(image, mask, class_names=self.class_names, ax=self.ax)

                title = "ID: {}\nImage file name: {}".format(
                    self.image_id,
                    os.path.basename(self.dataset.source_image_link(self.image_id))
                )

                self.fig.suptitle(title, fontsize=20)
                self.fig.canvas.draw_idle()
        
    def activate_text_box(self):
        self.text_box.set_active(True)
        self.text_box.label.set_text('Type in image id: ')

    def deactivate_text_box(self):
        self.text_box.set_active(False)
        self.text_box.label.set_text('Press enter to find by image ID')
        self.text_box.set_val(str(self.image_id))

    def text_box_callback(self, text):
        try:
            self.image_id = int(text)
            self.slider.set_val(self.image_id)
        except ValueError as e:
            print(e)

    def keypress_callback(self, event):
        if self.text_box.active:
            if event.key == 'enter':
                self.deactivate_text_box()
            else:
                pass
        else:
            if event.key == 'left':
                if self.image_id > 0:
                    self.image_id -= 1
                else:
                    self.image_id = self.dataset.num_images - 1
            elif self.dataset.num_images > 10 and event.key == 'up':
                if self.image_id > 10:
                    self.image_id -= 10
                else:
                    self.image_id += self.dataset.num_images - 10
            elif event.key == 'right':
                if self.image_id < self.dataset.num_images - 1:
                    self.image_id += 1
                else:
                    self.image_id = 0
            elif self.dataset.num_images > 10 and event.key == 'down':
                if self.image_id < self.dataset.num_images - 10:
                    self.image_id += 10
                else:
                    self.image_id -= self.dataset.num_images - 10
            elif event.key == 'enter':
                self.activate_text_box()
                return None
            elif event.key == 'escape':
                plt.close()
                return None
            else:
                print('Unknown Keyboard Input: {}'.format(event.key))
                return None

            self.text_box.set_val(str(self.image_id))
            self.slider.set_val(self.image_id)

    def slider_callback(self, val):
        self.image_id = int(self.slider.val)
        self.slider.valtext.set_text('{}'.format(self.image_id))
        self.text_box.set_val(str(self.image_id))
        self.display_image_and_label()



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

assert args.subset in win_titles,\
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
    dataset, win_titles[args.subset], args.mode)
plt.show()
