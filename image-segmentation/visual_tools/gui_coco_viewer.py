import os

from mask_rcnn.utils import extract_bboxes
from visual_tools.visualize import display_instances
from .gui_viewer import GuiViewer

class GuiCocoViewer(GuiViewer):
    def __init__(self, figurename, dataset):
        super(GuiCocoViewer, self).__init__(figurename)

        self.dataset = dataset

        self.num_images = self.dataset.num_images

        self.create_slider()
        self.create_textbox()

        self.display()

    def display(self):
        should_update = super(GuiCocoViewer, self).display()
        if should_update:
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
