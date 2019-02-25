from keras_model_wrapper import SemanticModelWrapper
from segmentation_models.unet.builder import build_unet

############################################################
#  Unet Class
############################################################

class Unet(SemanticModelWrapper):

    def build(self):
        super(Unet, self).build()

        backbone, feature_layers = self.get_backbone_and_feature_layers(4)

        unet = build_unet(backbone,
                          self.config.NUM_CLASSES - 1, # exclude the background
                          feature_layers,
                          decoder_filters=self.config.DECODER_FILTERS,
                          block_type=self.config.DECODER_BLOCK_TYPE,
                          activation=self.config.LAST_ACTIVATION,
                          n_upsample_blocks=len(self.config.DECODER_FILTERS),
                          upsample_rates=[2]*(len(feature_layers) + 1),
                          use_batchnorm=self.config.DECODER_USE_BATCHNORM
                          )

        return self.resolve_input_output(unet, 'unet')
