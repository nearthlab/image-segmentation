import os
import numpy as np

from data_generators.utils import load_image_rgb
from gui_viewer import GuiViewer

class GuiImageViewer(GuiViewer):
    def __init__(self, image_dir):
        super(GuiImageViewer, self).__init__('Image Viewer')
        assert os.path.isdir(image_dir), 'No such directory: {}'.format(image_dir)

        # Get the list of image files
        self.image_files = sorted([os.path.join(image_dir, x)
                            for x in os.listdir(image_dir)
                            if x.lower().endswith('.jpg') or x.lower().endswith('.png') or x.lower().endswith('.bmp')
                            ])
        self.num_images = len(self.image_files)

        self.create_slider()
        self.create_textbox()

        self.display()

    def display(self):
        should_update = super(GuiImageViewer, self).display()
        if should_update:
            image_file = self.image_files[self.image_id]
            print('Processing {}...'.format(os.path.basename(image_file)))
            img = load_image_rgb(image_file)
            self.ax.imshow(img.astype(np.uint8))

            title = "ID: {}\nImage file name: {}".format(
                self.image_id,
                os.path.basename(image_file)
            )
            self.fig.suptitle(title, fontsize=20)
            self.fig.canvas.draw_idle()


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Basic image viewer.')

    parser.add_argument('-i', '--image_dir', required=True,
                        metavar='/path/to/directory',
                        help='Path to a directory containing images')

    args = parser.parse_args()

    viewer = GuiImageViewer(args.image_dir)
    plt.show()