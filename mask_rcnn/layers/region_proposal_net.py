import tensorflow as tf
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import Lambda
from keras.layers import Input
from keras.models import Model

############################################################
#  Region Proposal Network (RPN)
############################################################

def rpn_graph(feature_map, anchors_per_location, anchor_stride):
    '''Builds the computation graph of Region Proposal Network.

    feature_map: backbone features [batch, height, width, depth]
    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).

    Returns:
        rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
        rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
        rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                  applied to anchors.
    '''
    # TODO: check if stride of 2 causes alignment issues if the feature map
    # is not even.
    # Shared convolutional base of the RPN
    # feature_map.shape == (B, H, W, D) -> shared.shape == (B, H, W, 512)
    shared = Conv2D(512, (3, 3), padding='same', activation='relu',
                       strides=anchor_stride,
                       name='rpn_conv_shared')(feature_map)

    # Anchor Score. [batch, height, width, anchors per location * 2].
    # feature_map.shape == (B, H, W, D) -> x.shape == (B, H, W, 2 * anchors_per_location)
    x = Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
                  activation='linear', name='rpn_class_raw')(shared)

    # Reshape to [batch, anchors, 2]
    # feature_map.shape == (B, H, W, D) -> rpn_class_logits.shape == (B, H * W * anchors_per_location, 2)
    rpn_class_logits = Lambda(
        lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

    # Softmax on last dimension of BG/FG.
    # feature_map.shape == (B, H, W, D) -> rpn_class_logits.shape == (B, H * W * anchors_per_location, 2)
    rpn_probs = Activation(
        'softmax', name='rpn_class_xxx')(rpn_class_logits)

    # Bounding box refinement. [batch, H, W, anchors per location * depth]
    # where depth is [x, y, log(w), log(h)]
    # feature_map.shape == (B, H, W, D) -> x.shape == (B, H, W, anchors per location * 4)
    x = Conv2D(anchors_per_location * 4, (1, 1), padding='valid',
                  activation='linear', name='rpn_bbox_pred')(shared)

    # Reshape to [batch, anchors, 4]
    # feature_map.shape == (B, H, W, D) -> x.shape == (B, H * W * anchors per location, 4)
    rpn_bbox = Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)

    return [rpn_class_logits, rpn_probs, rpn_bbox]


def RegionProposalNet(anchor_stride, anchors_per_location, depth):
    '''Builds a Keras model of the Region Proposal Network.
    It wraps the RPN graph so it can be used multiple times with shared
    weights.

    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).
    depth: Depth of the backbone feature map.

    Returns a Keras Model object. The model outputs, when called, are:
    rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
    rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
    rpn_bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                applied to anchors.
    '''
    input_feature_map = Input(shape=[None, None, depth],
                                 name='input_rpn_feature_map')
    outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
    return Model([input_feature_map], outputs, name='rpn_model')

