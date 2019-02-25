from keras_model_wrapper import SemanticModelWrapper
from segmentation_models.pspnet.builder import build_psp

############################################################
#  PSPNet Class
############################################################

def _get_layer_by_factor(feature_layers, factor):
    if factor == 4:
        return feature_layers[-1]
    elif factor == 8:
        return feature_layers[-2]
    elif factor == 16:
        return feature_layers[-3]
    else:
        raise ValueError('Unsupported factor - `{}`, Use 4, 8 or 16.'.format(factor))


def _shape_guard(factor, input_size):
    min_size = factor * 6

    res = (input_size % min_size != 0 or input_size < min_size)
    if res:
        raise ValueError('Wrong input size {}, input H and W should '.format(input_size) +
                         'be divisible by `{}`'.format(min_size))

class PSPNet(SemanticModelWrapper):

    def build(self):
        super(PSPNet, self).build()
        _shape_guard(self.config.DOWNSAMPLE_FACTOR, self.config.IMAGE_SIZE)

        backbone, feature_layers = self.get_backbone_and_feature_layers(3)

        psp_layers = _get_layer_by_factor(feature_layers, self.config.DOWNSAMPLE_FACTOR)

        pspnet = build_psp(backbone, psp_layers,
                           last_upsampling_factor=self.config.DOWNSAMPLE_FACTOR,
                           classes=self.config.NUM_CLASSES - 1,  # exclude the background
                           conv_filters=self.config.PSP_CONV_FILTERS,
                           pooling_type=self.config.PSP_POOLING_TYPE,
                           activation=self.config.LAST_ACTIVATION,
                           use_batchnorm=self.config.DECODER_USE_BATCHNORM,
                           dropout=self.config.DECODER_PYRAMID_DROPOUT,
                           final_interpolation=self.config.DECODER_FINAL_INTERPOLATION
                           )

        return self.resolve_input_output(pspnet, 'pspnet')
