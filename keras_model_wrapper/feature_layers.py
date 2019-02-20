# Copied DEFAULT_FEATURE_LAYERS and get_feature_layers
# from segmentation_models.backbone
# This is to avoid importing updated backbone model
# (https://github.com/qubvel/segmentation_models/issues/56)
DEFAULT_FEATURE_LAYERS = {

    # List of layers to take features from backbone in the following order:
    # (x16, x8, x4, x2, x1) - `x4` mean that features has 4 times less spatial
    # resolution (Height x Width) than input image.

    # VGG
    'vgg16': ('block5_conv3', 'block4_conv3', 'block3_conv3', 'block2_conv2', 'block1_conv2'),
    'vgg19': ('block5_conv4', 'block4_conv4', 'block3_conv4', 'block2_conv2', 'block1_conv2'),

    # ResNets
    'resnet18': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
    'resnet34': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
    'resnet50': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
    'resnet101': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
    'resnet152': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),

    # ResNeXt
    'resnext50': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
    'resnext101': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),

    # Inception
    'inceptionv3': (228, 86, 16, 9),
    'inceptionresnetv2': (594, 260, 16, 9),

    # DenseNet
    'densenet121': (311, 139, 51, 4),
    'densenet169': (367, 139, 51, 4),
    'densenet201': (479, 139, 51, 4),

    # SE models
    'seresnet18': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
    'seresnet34': ('stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'),
    'seresnet50': (233, 129, 59, 4),
    'seresnet101': (522, 129, 59, 4),
    'seresnet152': (811, 197, 59, 4),
    'seresnext50': (1065, 577, 251, 4),
    'seresnext101': (2442, 577, 251, 4),
    'senet154': (6837, 1614, 451, 12),

    # Mobile Nets
    'mobilenet': ('conv_pw_11_relu', 'conv_pw_5_relu', 'conv_pw_3_relu', 'conv_pw_1_relu'),
    'mobilenetv2': ('block_13_expand_relu', 'block_6_expand_relu', 'block_3_expand_relu', 'block_1_expand_relu'),

}

def get_feature_layers(name, n=5):
    return DEFAULT_FEATURE_LAYERS[name][:n]