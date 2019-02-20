import time
import argparse
import numpy as np

from keras.applications.imagenet_utils import decode_predictions
from classification_models import Classifiers

from data_generators.utils import resize, load_image_rgb

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Imagenet classification example.')
parser.add_argument('-b', '--backbone', required=True,
                    metavar='backbone model name',
                    help='The name of the backbone architecture')
args = parser.parse_args()

backbone = args.backbone
classifier, preprocess_input = Classifiers.get(backbone)

# load model
model = classifier(input_shape=(224, 224, 3), include_top=True, weights='imagenet')

image_files = ['cat1.jpg', 'cat2.jpg', 'dog1.jpg', 'dog2.jpg']
print('=============results=============')
for image_file in image_files:
    # read and prepare image
    img = load_image_rgb(image_file)
    x = resize(img, (224, 224), preserve_range=True)
    x = preprocess_input(x)
    x = np.expand_dims(x, 0)

    # processing image
    t = time.time()
    y = model.predict(x)
    time_taken = time.time() - t

    # result
    _, name, prob = decode_predictions(y)[0][0]
    print('{}: {}({})'.format(image_file, name, prob))
    print('\ttime taken: {}'.format(time_taken))

    # plt.imshow(img)
    # plt.title('{}: {}'.format(image_file, name))
    # plt.show()
print('=================================')
