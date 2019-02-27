import argparse

from data_generators.coco import CocoDataset, evaluate_coco

from models import MaskRCNN
from config import load_config

# Import your custom backbone
from matterport_resnet import ResNet50, ResNet101, preprocess_input

# You can add or override existing backbone model
# and corresponding preprocessing function as follows:
from classification_models import Classifiers
Classifiers._models.update({
    'resnet50': [ResNet50, preprocess_input],
    'resnet101': [ResNet101, preprocess_input],
})

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Evaluate Mask R-CNN detector.')

    parser.add_argument('-w', '--weights', required=False,
                        default='MaskRCNN_coco.h5',
                        metavar='/path/to/MaskRCNN_coco.h5',
                        help='Path to maskrcnn weights.h5 file')
    parser.add_argument('-c', '--model_cfg', required=False,
                        default='MaskRCNN_coco.cfg',
                        metavar='/path/to/MaskRCNN_coco.cfg',
                        help='Path to MaskRCNN_coco.cfg file')
    parser.add_argument('-d', '--dataset', required=True,
                        metavar='/path/to/coco/',
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--tag', required=False,
                        default='2017',
                        metavar='<tag>',
                        help='Tag of the MS-COCO dataset (default=2017)')
    parser.add_argument('-e', '--eval_type', required=True,
                        default='bbox',
                        metavar="<evaluation type>",
                        help='"bbox" or "segm" for bounding box or segmentation evaluation')
    parser.add_argument('-l', '--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')

    args = parser.parse_args()

    assert args.eval_type in ['bbox', 'segm'], 'Invalid evaluation type {}'.format(args.eval_type)

    dataset_val = CocoDataset()
    coco = dataset_val.load_coco(args.dataset, 'val', tag=args.tag, return_coco=True)
    dataset_val.prepare()

    class_names = {0: 'background', 1: 'blade'}

    # Model Configurations
    model_config = load_config(args.model_cfg)

    # Create model
    print('Building MaskRCNN model...')
    maskrcnn = MaskRCNN(config=model_config)
    print('Loading {}...'.format(args.weights))
    maskrcnn.load_weights(args.weights)

    evaluate_coco(maskrcnn, dataset_val, coco, args.eval_type, limit=int(args.limit))