import time
from tqdm import tqdm
import argparse
import numpy as np

from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
from data_generators.coco_dataset import CocoDataset

from models import MaskRCNN
from config import load_config

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    image_ids = image_ids.tolist()
    print("Running COCO evaluation on {} images.".format(len(image_ids)))

    for i, image_id in enumerate(tqdm(image_ids)):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.predict(image)
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.params.areaRng = [[0 ** 2, 512 ** 2], [0 ** 2, 128 ** 2], [128 ** 2, 256 ** 2], [256 ** 2, 512 ** 2]]
    cocoEval.params.areaRngLbl = ['all', 'small', 'medium', 'large']
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Evaluate Mask R-CNN detector.')

    parser.add_argument('-w', '--weights', required=True,
                        metavar='/path/to/weights.h5',
                        help='Path to maskrcnn weights.h5 file')
    parser.add_argument('-c', '--model_cfg', required=True,
                        metavar='/path/to/model.cfg',
                        help='Path to model.cfg file')
    parser.add_argument('-d', '--dataset', required=True,
                        metavar='/path/to/coco/',
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--tag', required=False,
                        default='v1',
                        metavar='<tag>',
                        help='Tag of the MS-COCO dataset (v1 or v2) (default=v1)')
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