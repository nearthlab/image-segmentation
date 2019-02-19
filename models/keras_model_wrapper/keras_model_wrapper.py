import os
from abc import *

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

class KerasModelWrapper(metaclass=ABCMeta):
    def __init__(self, config):
        self.branched_config = config
        self.config = config.flatten()
        self.branched_config.display()
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in self.config.GPU_IDS])
        self.backbone_layer_names = []
        self.model = self.build()
        assert len(self.backbone_layer_names) > 0,\
            'Implementation error: When you override self.build() function, \
            you must provide backbone layer names in the attribute self.backbone_layer_names'
        # Add multi-GPU support.
        if len(self.config.GPU_IDS) > 1:
            from .parallel_model import ParallelModel
            self.model = ParallelModel(self.model, self.config.GPU_IDS)

    @abstractmethod
    def build(self):
        assert self.config.MODE in ['training', 'inference']

    @abstractmethod
    def predict(self, image, threshold=0.5):
        assert self.config.MODE == 'inference'

    def load_weights(self, model_path, by_name=True, exclude=None):
        '''Modified version of the corresponding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exclude: list of layer names to exclude
        '''
        import h5py
        from keras.engine import saving

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(model_path, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = self.model.inner_model.layers if hasattr(self.model, 'inner_model') \
            else self.model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            saving.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            saving.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

    def save_weights(self, filepath, overwrite=True):
        self.model.save_weights(filepath=filepath, overwrite=overwrite)

