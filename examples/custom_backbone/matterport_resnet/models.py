from .builder import build_resnet
from classification_models.utils import load_model_weights


weights_collection = [
    # The implementation of ResNet in matterport/Mask_RCNN
    # Original implementation: https://github.com/fchollet/deep-learning-models
    {
        'model': 'resnet50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': True,
        'url': 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5',
        'name': 'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
        'md5': 'a7b3fe01876f51b976af0dea6bc144eb',
    },

    {
        'model': 'resnet50',
        'dataset': 'imagenet',
        'classes': 1000,
        'include_top': False,
        'url': 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
        'name': 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
        'md5': 'a268eb855778b3df3c7506639542a6af',
    }
]

def ResNet50(input_shape, input_tensor=None, weights=None,
             classes=1000, include_top=True):
    model = build_resnet(input_tensor=input_tensor,
                         input_shape=input_shape,
                         pooling=None,
                         block_count=5,
                         classes=classes,
                         include_top=include_top)

    model.name = 'resnet50'

    if weights:
        load_model_weights(weights_collection, model, weights, classes, include_top)
    return model

def ResNet101(input_shape, input_tensor=None, weights=None,
              classes=1000, include_top=True):
    model = build_resnet(input_tensor=input_tensor,
                         input_shape=input_shape,
                         pooling=None,
                         block_count=22,
                         classes=classes,
                         include_top=include_top)
    model.name = 'resnet101'

    if weights:
        raise NotImplemented

    return model

