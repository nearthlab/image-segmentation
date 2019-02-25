from abc import *
import numpy as np

import keras.backend as K
from keras.models import Model
from keras.layers import Lambda, Input
from keras.losses import binary_crossentropy

from keras_model_wrapper import KerasModelWrapper

from data_generators.utils import resize

from classification_models import Classifiers
from .feature_layers import get_feature_layers
from segmentation_models.losses import jaccard_loss as jaccard_loss_graph
from segmentation_models.losses import dice_loss as dice_loss_graph


def bce_loss_graph(gt, pr):
    return K.mean(binary_crossentropy(gt, pr))
    # return K.mean(w * K.clip(binary_crossentropy(gt, pr), K.epsilon(), 1.0))

############################################################
#  Semantic Segmentation Model Class
############################################################

class SemanticModelWrapper(KerasModelWrapper, metaclass=ABCMeta):

    @abstractmethod
    def build(self):
        super(SemanticModelWrapper, self).build()


    def get_backbone_and_feature_layers(self, num_feature_layers):
        super(SemanticModelWrapper, self).build()
        image_shape = (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE, 3)
        classifier, self.preprocess_input = Classifiers.get(self.config.BACKBONE)
        backbone = classifier(input_shape=image_shape,
                                input_tensor=None,
                                weights=self.config.BACKBONE_WEIGHTS,
                                include_top=False)

        for layer in backbone.layers:
            self.backbone_layer_names.append(layer.name)

        if self.config.FEATURE_LAYERS == 'default':
            feature_layers = get_feature_layers(self.config.BACKBONE, n=num_feature_layers)
        else:
            feature_layers = self.config.FEATURE_LAYERS

        return backbone, feature_layers


    def resolve_input_output(self, base_model, name):
        if self.config.MODE == 'training':
            input_gt_masks = Input(
                shape=[self.config.IMAGE_SIZE, self.config.IMAGE_SIZE, None],
                name='input_gt_masks', dtype=float)

            bce_loss = Lambda(lambda x: bce_loss_graph(*x), name='bce_loss') \
                ([input_gt_masks, base_model.output])

            jaccard_loss = Lambda(lambda x: jaccard_loss_graph(*x), name='jaccard_loss') \
                ([input_gt_masks, base_model.output])

            dice_loss = Lambda(lambda x: dice_loss_graph(*x), name='dice_loss') \
                ([input_gt_masks, base_model.output])

            inputs = base_model.inputs
            inputs += [input_gt_masks]
            outputs = base_model.outputs
            outputs += [bce_loss, jaccard_loss, dice_loss]
            model = Model(inputs, outputs, name=name)

            return model
        else:
            base_model.name = name
            return base_model


    def predict(self, image, threshold=0.5):
        '''Runs the detection pipeline.
        images: input image

        Returns a list that contains masks (float array with values 0.0 ~ 1.0) for each class
        '''
        super(SemanticModelWrapper, self).predict(image, threshold)

        height, width = image.shape[:2]
        if image.shape != (self.config.IMAGE_SIZE, self.config.IMAGE_SIZE, 3):
            image = resize(image, output_shape=(self.config.IMAGE_SIZE, self.config.IMAGE_SIZE), preserve_range=True)
        input = self.preprocess_input(image)
        input = np.expand_dims(input, axis=0).astype(np.float32)
        res = self.model.predict(input, batch_size=1)[0]

        num_channels = res.shape[-1]
        final_result = np.zeros((height, width, num_channels))
        for i in range(num_channels):
            resized_mask = resize(res[:, :, i], (height, width))
            resized_mask[resized_mask > threshold] = 1.0
            resized_mask[resized_mask <= threshold] = 0.0
            final_result[:, :, i] = resized_mask

        return final_result
