from keras.backend import is_keras_tensor
from keras.engine import get_source_inputs
from keras.models import Model

from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import ZeroPadding2D
from keras.layers import MaxPooling2D
from keras.layers import Input
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout

from .blocks import identity_block
from .blocks import conv_block

import keras
from distutils.version import StrictVersion

if StrictVersion(keras.__version__) < StrictVersion('2.2.0'):
    from keras.applications.imagenet_utils import _obtain_input_shape
else:
    from keras_applications.imagenet_utils import _obtain_input_shape

############################################################
#  Resnet Graph
############################################################

# Code adopted from:
# https://github.com/matterport/Mask_RCNN/blob/4f440de25a0fbb3f7c8998c5490163699ba5aa19/mrcnn/model.py#L171

def build_resnet(block_count=5,
     include_top=True,
     input_tensor=None,
     input_shape=None,
     pooling=None,
     classes=1000):

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=197,
                                      data_format='channels_last',
                                      require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape, name='data')
    else:
        if not is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Stage 0
    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    # Stage 1
    x = conv_block(x, 3, [64, 64, 256], stage=1, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=1, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=1, block='c')

    # Stage 2
    x = conv_block(x, 3, [128, 128, 512], stage=2, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=2, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=2, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=2, block='d')

    # Stage 3
    x = conv_block(x, 3, [256, 256, 1024], stage=3, block='a')
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=3, block=chr(98 + i))

    # Stage 4
    x = conv_block(x, 3, [512, 512, 2048], stage=4, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=4, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=4, block='c')

    # resnet top
    if include_top:
        x = AveragePooling2D((7, 7), name='avg_pool')(x)
        x = Dropout(0.3)(x)
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, x)

    return model

