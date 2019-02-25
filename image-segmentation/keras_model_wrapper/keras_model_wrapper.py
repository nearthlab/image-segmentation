import os
from abc import *

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
        assert self.config.BATCH_SIZE == 1

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

