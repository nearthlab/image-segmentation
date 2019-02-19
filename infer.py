import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from config import load_config
from models import get_model_wrapper
from data_generators.utils import load_image_rgb
from visual_tools.visualize import display_instances, display_segmentation
from visual_tools.gui_viewer import GuiViewer

class GuiInferenceViewer(GuiViewer):
    def __init__(self, image_dir, config_path, weights_path, threshold):
        super(GuiInferenceViewer, self).__init__('Inference Viewer')
        assert os.path.isdir(image_dir), 'No such directory: {}'.format(image_dir)

        self.threshold = threshold
        # Get the list of image files
        self.image_files = sorted([os.path.join(image_dir, x)
                            for x in os.listdir(image_dir)
                            if x.lower().endswith('.jpg') or x.lower().endswith('.png') or x.lower().endswith('.bmp')
                            ])
        self.num_images = len(self.image_files)

        self.model = get_model_wrapper(load_config(config_path))
        if weights_path:
            self.model.load_weights(weights_path)
        else:
            print('No weights path provided. Will use randomly initialized weights')

        self.create_slider()
        self.create_textbox()

        self.display()

    def display(self):
        should_update = super(GuiInferenceViewer, self).display()
        if should_update:
            image_file = self.image_files[self.image_id]
            print('Processing {}...'.format(os.path.basename(image_file)))
            img = load_image_rgb(image_file)
            t = time.time()
            det = self.model.predict(img.astype(np.float32), self.threshold)
            time_taken = time.time() - t
            print("\ttime taken: {}".format(time_taken))

            if self.model.config.MODEL == 'maskrcnn':

                display_instances(image=img, boxes=det['rois'],
                                  masks=det['masks'], class_ids=det['class_ids'],
                                  scores=det['scores'], title=image_file,
                                  ax=self.ax)
            else:

                display_segmentation(image=img, masks=det, ax=self.ax)

            title = "ID: {}\nImage file name: {}".format(
                self.image_id,
                os.path.basename(image_file)
            )
            self.fig.suptitle(title, fontsize=20)
            self.fig.canvas.draw_idle()


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
    parser.add_argument('-t', '--threshold', required=False,
                        type=float,
                        default=0.5,
                        metavar='Threshold value for inference',
                        help='Must be between 0 and 1.')
    args = parser.parse_args()

    assert args.threshold >= 0.0 and args.threshold < 1.0, \
        'Invalid threshold value {} given'.format(args.threshold)

    viewer = GuiInferenceViewer(args.image_dir, args.model_cfg, args.weights, args.threshold)
    plt.show()
