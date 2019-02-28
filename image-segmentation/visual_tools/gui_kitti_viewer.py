from visual_tools.visualize import display_segmentation
from .gui_viewer import GuiViewer

class GuiKittiViewer(GuiViewer):
    def __init__(self, figurename, dataset):
        super(GuiKittiViewer, self).__init__(figurename)

        self.dataset = dataset

        self.num_images = self.dataset.num_images

        self.create_slider()
        self.create_textbox()

        self.display()

    def display(self):
        should_update = super(GuiKittiViewer, self).display()
        if should_update:
            image = self.dataset.load_image(self.image_id)
            masks = self.dataset.load_mask(self.image_id)

            display_segmentation(image, masks, self.dataset.class_names, ax=self.ax)

            title = "ID: {}\nImage file name: {}".format(
                self.image_id,
                self.dataset.image_files[self.image_id]
            )

            self.fig.suptitle(title, fontsize=20)
            self.fig.canvas.draw_idle()
