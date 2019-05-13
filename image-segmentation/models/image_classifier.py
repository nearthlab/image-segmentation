import keras.backend as K
import tensorflow as tf

from keras.models import Model
from keras.layers import Lambda, Input

from classification_models import Classifiers
from keras_model_wrapper import KerasModelWrapper


def weighted_focal_loss(weights, gamma):
    weights = K.variable([weights])
    gamma = K.variable([gamma])

    def loss(gt, pr):
        # scale preds so that the class probas of each sample sum to 1
        pr /= tf.reduce_sum(pr, axis=-1, keep_dims=True)
        # manual computation of crossentropy
        pr = tf.clip_by_value(pr, K.epsilon(), 1. - K.epsilon())
        return K.mean(-tf.reduce_sum(gt * K.pow(1. - pr, gamma) * tf.log(pr) * weights, axis=-1))

    return loss


class ImageClassifier(KerasModelWrapper):

    def build(self):
        super(ImageClassifier, self).build()

        classifier, self.preprocess_input = Classifiers.get(self.config.BACKBONE)
        base_model = classifier(input_shape=self.config.IMAGE_SHAPE,
                           include_top=True, classes=self.config.NUM_CLASSES)

        for layer in base_model.layers:
            self.backbone_layer_names.append(layer.name)



        if self.config.MODE == 'training':
            input_label = Input(
                shape=[self.config.NUM_CLASSES],
                name='input_label', dtype=float)

            wfocal_loss_graph = weighted_focal_loss(self.config.CCE_WEIGHTS, self.config.FOCAL_LOSS_GAMMA)
            wfocal_loss = Lambda(lambda x: wfocal_loss_graph(*x), name='wfocal_loss') \
                ([input_label, base_model.output])

            inputs = base_model.inputs
            inputs += [input_label]
            outputs = base_model.outputs
            outputs += [wfocal_loss]
            model = Model(inputs, outputs, name=self.config.BACKBONE)

            return model
        else:
            return base_model


    def predict(self, image, threshold=0.5):
        super(ImageClassifier, self).predict(image, threshold)

        input = self.preprocess_input(image)

        return self.model.predict(input, batch_size=self.config.BATCH_SIZE)
