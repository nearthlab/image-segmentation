from models.keras_model_wrapper import SemanticModelWrapper
from segmentation_models.linknet.builder import build_linknet

############################################################
#  Linknet Class
############################################################

class Linknet(SemanticModelWrapper):

    def build(self):
        super(Linknet, self).build()

        backbone, feature_layers = self.get_backbone_and_feature_layers(4)

        linknet = build_linknet(backbone,
                                self.config.NUM_CLASSES - 1,  # exclude the background
                                feature_layers,
                                decoder_filters=self.config.DECODER_FILTERS,
                                upsample_layer=self.config.DECODER_BLOCK_TYPE,
                                activation=self.config.LAST_ACTIVATION,
                                n_upsample_blocks=len(self.config.DECODER_FILTERS),
                                upsample_rates=[2] * (len(feature_layers) + 1),
                                upsample_kernel_size=(3, 3),
                                use_batchnorm=self.config.DECODER_USE_BATCHNORM
                                )

        return self.resolve_input_output(linknet, 'linknet')
