from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import BatchNormalization
from keras.layers import TimeDistributed
from keras.layers import Activation

from .roi_align_layer import PyramidROIAlign

def mrcnn_mask_graph(rois, feature_maps, image_meta, pool_size, num_classes):
    '''Builds the computation graph of the mask head of Feature Pyramid Network.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers

    Returns: Masks [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
    '''
    # ROI Pooling
    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x = PyramidROIAlign([pool_size, pool_size],
                        name='roi_align_mask')([rois, image_meta] + feature_maps)

    # Conv layers
    x = TimeDistributed(Conv2D(256, (3, 3), padding='same'),
                           name='mrcnn_mask_conv1')(x)
    x = TimeDistributed(BatchNormalization(),
                           name='mrcnn_mask_bn1')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(256, (3, 3), padding='same'),
                           name='mrcnn_mask_conv2')(x)
    x = TimeDistributed(BatchNormalization(),
                           name='mrcnn_mask_bn2')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(256, (3, 3), padding='same'),
                           name='mrcnn_mask_conv3')(x)
    x = TimeDistributed(BatchNormalization(),
                           name='mrcnn_mask_bn3')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(256, (3, 3), padding='same'),
                           name='mrcnn_mask_conv4')(x)
    x = TimeDistributed(BatchNormalization(),
                           name='mrcnn_mask_bn4')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2DTranspose(256, (2, 2), strides=2, activation='relu'),
                           name='mrcnn_mask_deconv')(x)
    x = TimeDistributed(Conv2D(num_classes, (1, 1), strides=1, activation='sigmoid'),
                           name='mrcnn_mask')(x)
    return x