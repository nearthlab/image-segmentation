from .trainer import Trainer
from .maskrcnn import MaskRCNN
from .unet import Unet
from .fpn import FPN
from .linknet import Linknet
from .pspnet import PSPNet

MODEL_WRAPPERS = {
    'maskrcnn': MaskRCNN,
    'unet': Unet,
    'fpn': FPN,
    'linknet': Linknet,
    'pspnet': PSPNet
}

def get_model_wrapper(config):
    model_name = config.MODEL if isinstance(config.MODEL, str) else config.MODEL.MODEL
    return MODEL_WRAPPERS.get(model_name)(config)
