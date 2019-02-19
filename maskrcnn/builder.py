import numpy as np
import tensorflow as tf

from keras.models import Model

from keras.layers import Input
from keras.layers import Lambda
from keras.layers import Concatenate

from keras.backend import shape

#from segmentation_models.backbones import get_backbone, DEFAULT_FEATURE_LAYERS
from classification_models import Classifiers
from models.keras_model_wrapper import get_feature_layers
from segmentation_models.utils import get_layer_number

from .utils import norm_boxes_graph, get_anchors
from data_generators.utils import parse_image_meta_graph

from .layers import DetectionLayer, DetectionTargetLayer, fpn_graph,\
    RegionProposalNet, ProposalLayer, mrcnn_mask_graph, mrcnn_classifier_graph

from .losses import rpn_class_loss_graph, rpn_bbox_loss_graph,\
    mrcnn_class_loss_graph, mrcnn_bbox_loss_graph, mrcnn_mask_loss_graph


def get_default_feature_layers(config, backbone):
    try:
        feature_layers = []
        # We want our feature layers to be listed in decreasing output size (width, height)
        for l in reversed(get_feature_layers(config.BACKBONE)[:-1]):
            feature_layers.append(l)
        # and we use the last backbone layer output instead of the last layer in DEFAULT_FEATURE_LAYERS
        feature_layers.append(backbone.layers[-1].name)
        return feature_layers
    except KeyError:
        print('No default feature layers for {} provided yet.',
              'Please specify the layer names (or indexes) explicitly in config file'.format(config.BACKBONE))
        exit(1)


def build_maskrcnn(config):
    # Image size must be dividable by 2 multiple times
    if config.IMAGE_SIZE % 64 != 0:
        raise Exception('Image size must be dividable by 2 at least 6 times '
                        'to avoid fractions when downscaling and upscaling.'
                        'For example, use 256, 320, 384, 448, 512, ... etc. ')

    image_shape = (config.IMAGE_SIZE, config.IMAGE_SIZE, 3)
    # Inputs
    input_image = Input(
        shape=image_shape, name='input_image')
    input_image_meta = Input(shape=[config.IMAGE_META_SIZE],
                             name='input_image_meta')

    classifier = Classifiers.get_classifier(config.BACKBONE)
    backbone = classifier(input_tensor=input_image,
                          input_shape=image_shape,
                          include_top=False,
                          weights=config.BACKBONE_WEIGHTS)

    backbone_layer_names = []
    for layer in backbone.layers:
        backbone_layer_names.append(layer.name)

    if config.FEATURE_LAYERS == 'default':
        feature_layers = get_default_feature_layers(config, backbone)
    else:
        feature_layers = config.FEATURE_LAYERS

    # convert layer names to indices
    skip_connection_idx = ([get_layer_number(backbone, l) if isinstance(l, str) else l
                            for l in feature_layers])
    backbone_features = []
    for idx in skip_connection_idx:
        backbone_features.append(backbone.layers[idx].output)

    feature_pyramids = fpn_graph(backbone_features, config.TOP_DOWN_PYRAMID_SIZE)
    rpn_feature_maps = feature_pyramids
    mrcnn_feature_maps = feature_pyramids[:-1]

    # Anchors
    if config.MODE == 'training':
        anchors = get_anchors(config)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
        # A hack to get around Keras's bad support for constants
        anchors = Lambda(lambda x: tf.Variable(anchors), name='anchors')(input_image)
    else:
        # Anchors in normalized coordinates
        anchors = Input(shape=[None, 4], name='input_anchors')

    # RPN Model
    rpn = RegionProposalNet(config.RPN_ANCHOR_STRIDE,
                            len(config.RPN_ANCHOR_RATIOS), config.TOP_DOWN_PYRAMID_SIZE)

    # Loop through FPN outputs
    layer_outputs = []  # list of lists
    for p in rpn_feature_maps:
        layer_outputs.append(rpn([p]))

    # Concatenate layer outputs
    # Convert from list of lists of level outputs to list of lists
    # of outputs across levels.
    # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
    output_names = ['rpn_class_logits', 'rpn_class', 'rpn_bbox']
    outputs = list(zip(*layer_outputs))
    outputs = [Concatenate(axis=1, name=n)(list(o))
               for o, n in zip(outputs, output_names)]

    '''
    input: (B, N, N, 3)
    anchors_per_location: len(config.RPN_ANCHOR_RATIOS)
    the coef: 341 / 4096 = (1/4)**2 + (1/8)**2 + (1/16)**2 + (1/32)**2 + (1/64)**2
        rpn_class_logits: (B, anchors_per_location * N * N * (341 / 4096), 2)
        rpn_class: (B, anchors_per_location * N * N * (341 / 4096), 2)
        rpn_bbox: (B, anchors_per_location * N * N * (341 / 4096), 4)

    '''
    rpn_class_logits, rpn_class, rpn_bbox = outputs

    # Generate proposals
    # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
    # and zero padded.
    proposal_count = config.POST_NMS_ROIS_TRAINING if config.MODE == 'training' \
        else config.POST_NMS_ROIS_INFERENCE
    rpn_rois = ProposalLayer(
        proposal_count=proposal_count,
        nms_threshold=config.RPN_NMS_THRESHOLD,
        name='ROI',
        config=config)([rpn_class, rpn_bbox, anchors])

    if config.MODE == 'training':
        # Create all the input layers
        # RPN GT
        input_rpn_match = Input(
            shape=[None, 1], name='input_rpn_match', dtype=tf.int32)
        input_rpn_bbox = Input(
            shape=[None, 4], name='input_rpn_bbox', dtype=tf.float32)

        # Detection GT (class IDs, bounding boxes, and masks)
        # 1. GT Class IDs (zero padded)
        input_gt_class_ids = Input(
            shape=[None], name='input_gt_class_ids', dtype=tf.int32)
        # 2. GT Boxes in pixels (zero padded)
        # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
        input_gt_boxes = Input(
            shape=[None, 4], name='input_gt_boxes', dtype=tf.float32)
        # Normalize coordinates
        gt_boxes = Lambda(lambda x: norm_boxes_graph(
            x, shape(input_image)[1:3]))(input_gt_boxes)
        # 3. GT Masks (zero padded)
        # [batch, height, width, MAX_GT_INSTANCES]
        if config.USE_MINI_MASK:
            input_gt_masks = Input(
                shape=[config.MINI_MASK_SHAPE[0],
                       config.MINI_MASK_SHAPE[1], None],
                name='input_gt_masks', dtype=bool)
        else:
            input_gt_masks = Input(
                shape=[image_shape[0], image_shape[1], None],
                name='input_gt_masks', dtype=bool)

        # Class ID mask to mark class IDs supported by the dataset the image
        # came from.
        active_class_ids = Lambda(
            lambda x: parse_image_meta_graph(x)['active_class_ids']
        )(input_image_meta)

        # Generate detection targets
        # Subsamples proposals and generates target outputs for training
        # Note that proposal class IDs, gt_boxes, and gt_masks are zero
        # padded. Equally, returned rois and targets are zero padded.
        rois, target_class_ids, target_bbox, target_mask = \
            DetectionTargetLayer(config, name='proposal_targets')([
                rpn_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

        # Network Heads
        # TODO: verify that this handles zero padded ROIs
        mrcnn_class_logits, mrcnn_class, mrcnn_bbox = \
            mrcnn_classifier_graph(rois, mrcnn_feature_maps, input_image_meta,
                                   config.POOL_SIZE, config.NUM_CLASSES,
                                   fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

        mrcnn_mask = mrcnn_mask_graph(rois, mrcnn_feature_maps,
                                      input_image_meta,
                                      config.MASK_POOL_SIZE,
                                      config.NUM_CLASSES)

        # TODO: clean up (use tf.identify if necessary)
        output_rois = Lambda(lambda x: x * 1, name='output_rois')(rois)

        # Losses
        rpn_class_loss = Lambda(lambda x: rpn_class_loss_graph(*x), name='rpn_class_loss')(
            [input_rpn_match, rpn_class_logits])
        rpn_bbox_loss = Lambda(lambda x: rpn_bbox_loss_graph(config, *x), name='rpn_bbox_loss')(
            [input_rpn_bbox, input_rpn_match, rpn_bbox])
        class_loss = Lambda(lambda x: mrcnn_class_loss_graph(*x), name='mrcnn_class_loss')(
            [target_class_ids, mrcnn_class_logits, active_class_ids])
        bbox_loss = Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name='mrcnn_bbox_loss')(
            [target_bbox, target_class_ids, mrcnn_bbox])
        mask_loss = Lambda(lambda x: mrcnn_mask_loss_graph(*x), name='mrcnn_mask_loss')(
            [target_mask, target_class_ids, mrcnn_mask])

        # Model
        inputs = [input_image, input_image_meta,
                  input_rpn_match, input_rpn_bbox, input_gt_class_ids, input_gt_boxes, input_gt_masks]
        outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                   mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
                   rpn_rois, output_rois,
                   rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]
        model = Model(inputs, outputs, name='maskrcnn')
    else:
        # Network Heads
        # Proposal classifier and BBox regressor heads
        mrcnn_class_logits, mrcnn_class, mrcnn_bbox = \
            mrcnn_classifier_graph(rpn_rois, mrcnn_feature_maps, input_image_meta,
                                   config.POOL_SIZE, config.NUM_CLASSES,
                                   fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

        # Detections
        # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in
        # normalized coordinates
        detections = DetectionLayer(config, name='mrcnn_detection')(
            [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

        # Create masks for detections
        detection_boxes = Lambda(lambda x: x[..., :4])(detections)
        mrcnn_mask = mrcnn_mask_graph(detection_boxes, mrcnn_feature_maps,
                                      input_image_meta,
                                      config.MASK_POOL_SIZE,
                                      config.NUM_CLASSES)

        model = Model([input_image, input_image_meta, anchors],
                      [detections, mrcnn_class, mrcnn_bbox,
                       mrcnn_mask, rpn_rois, rpn_class, rpn_bbox],
                      name='maskrcnn')

    return model, backbone_layer_names