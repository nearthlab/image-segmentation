import os
import time
import numpy as np

from config import load_config
from models import get_model_wrapper
from data_generators.utils import load_image_rgb
from visual_tools.visualize import display_instances, display_segmentation
from .gui_viewer import GuiViewer

class GuiInferenceViewer(GuiViewer):
    def __init__(self, image_dir, config_path, weights_path, threshold=0.5, class_names=None):
        super(GuiInferenceViewer, self).__init__('Inference Viewer')
        assert os.path.isdir(image_dir), 'No such directory: {}'.format(image_dir)

        self.threshold = threshold
        self.class_names = class_names
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
                                  class_names=self.class_names, ax=self.ax)
            else:

                display_segmentation(image=img, masks=det, class_names=self.class_names, ax=self.ax)

            title = "ID: {}\nImage file name: {}".format(
                self.image_id,
                os.path.basename(image_file)
            )
            self.fig.suptitle(title, fontsize=20)
            self.fig.canvas.draw_idle()
