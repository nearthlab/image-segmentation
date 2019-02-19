from abc import *

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.widgets import TextBox

class GuiViewer(metaclass=ABCMeta):
    def __init__(self, figurename):
        self.fig, self.ax = plt.subplots(1, figsize=(16, 16), num=figurename)
        self.image_id = 0
        self.prev_image_id = None
        self.num_images = 0

        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)

        self.fig.canvas.mpl_connect('key_press_event', self.keypress_callback)

        self.slider = None
        self.text_box = None

    @abstractmethod
    def display(self):
        assert self.num_images > 0, 'No image to display'
        if self.prev_image_id == self.image_id:
            return False
        else:
            self.ax.clear()
            print('image id: {}'.format(self.image_id))
            self.prev_image_id = self.image_id
            return True

    def create_slider(self):
        self.axcolor = 'lightgoldenrodyellow'
        ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor=self.axcolor)
        self.slider = Slider(ax_slider, 'Id', 0, self.num_images - 1, valinit=0)
        self.slider.on_changed(self.slider_callback)

    def create_textbox(self):
        axbox = plt.axes([0.2, 0.05, 0.65, 0.03])
        self.text_box = TextBox(axbox, '', initial=str(self.image_id))
        self.text_box.on_submit(self.text_box_callback)
        self.deactivate_text_box()

    def activate_text_box(self):
        self.text_box.set_active(True)
        self.text_box.label.set_text('Type in image id: ')

    def deactivate_text_box(self):
        self.text_box.set_active(False)
        self.text_box.label.set_text('Press enter to find by image ID')
        self.text_box.set_val(str(self.image_id))

    def text_box_callback(self, text):
        try:
            self.image_id = min(self.num_images - 1, int(text))
            if self.slider:
                self.slider.set_val(self.image_id)
            self.display()
        except ValueError as e:
            print(e)

    def keypress_callback(self, event):
        if self.text_box and self.text_box.active:
            if event.key == 'enter':
                self.deactivate_text_box()
            else:
                pass
        else:
            if event.key == 'left':
                if self.image_id > 0:
                    self.image_id -= 1
                else:
                    self.image_id = self.num_images - 1
            elif self.num_images > 10 and event.key == 'up':
                if self.image_id > 10:
                    self.image_id -= 10
                else:
                    self.image_id += self.num_images - 10
            elif event.key == 'right':
                if self.image_id < self.num_images - 1:
                    self.image_id += 1
                else:
                    self.image_id = 0
            elif self.num_images > 10 and event.key == 'down':
                if self.image_id < self.num_images - 10:
                    self.image_id += 10
                else:
                    self.image_id -= self.num_images - 10
            elif event.key == 'enter':
                self.activate_text_box()
                return None
            elif event.key == 'escape':
                plt.close()
                return None
            else:
                print('Unknown Keyboard Input: {}'.format(event.key))
                return None

            self.display()
            if self.slider:
                self.slider.set_val(self.image_id)
            if self.text_box:
                self.text_box.set_val(str(self.image_id))

    def slider_callback(self, val):
        self.image_id = int(self.slider.val)
        self.slider.valtext.set_text('{}'.format(self.image_id))
        if self.text_box:
            self.text_box.set_val(str(self.image_id))
        self.display()
