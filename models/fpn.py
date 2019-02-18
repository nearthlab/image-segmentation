from models.keras_model_wrapper import SemanticModelWrapper
from segmentation_models.fpn.builder import build_fpn

############################################################
#  FPN Class
############################################################

class FPN(SemanticModelWrapper):

    def build(self):
        super(FPN, self).build()

        backbone, feature_layers = self.get_backbone_and_feature_layers(3)

        fpn = build_fpn(backbone, feature_layers,
                        classes=self.config.NUM_CLASSES - 1, # exclude the background
                        pyramid_filters=self.config.DECODER_PYRAMID_BLOCK_FILTERS,
                        segmentation_filters=self.config.DECODER_PYRAMID_BLOCK_FILTERS // 2,
                        upsample_rates=[2]*len(feature_layers),
                        use_batchnorm=self.config.DECODER_USE_BATCHNORM,
                        dropout=self.config.DECODER_PYRAMID_DROPOUT,
                        last_upsample=2 ** (5 - len(feature_layers)),
                        interpolation=self.config.DECODER_FINAL_INTERPOLATION,
                        activation=self.config.LAST_ACTIVATION
                        )

        return self.resolve_input_output(fpn, 'fpn')
