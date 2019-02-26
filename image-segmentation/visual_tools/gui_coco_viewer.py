import os

from data_generators.semantic import load_image_gt
from mask_rcnn.utils import extract_bboxes
from visual_tools.visualize import display_instances, display_segmentation
from .gui_viewer import GuiViewer

class GuiCocoViewer(GuiViewer):
    def __init__(self, figurename, dataset, mode):
        super(GuiCocoViewer, self).__init__(figurename)
        assert mode in ['instance', 'semantic']
        self.mode = mode

        self.dataset = dataset
        if self.mode == 'semantic':
            self.class_names = [''] * self.dataset.num_classes

            for i, info in enumerate(self.dataset.class_info):
                self.class_names[i] = info['name']

            if 'BG' in self.class_names:
                self.class_names.remove('BG')

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
