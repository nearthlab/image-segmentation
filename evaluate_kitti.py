import numpy as np
import argparse

from tqdm import tqdm
from config import load_config
from models import get_model_wrapper
from data_generators.kitti import load_image_gt, KittiDataset

def compute_confusion_matrix(gt_mask, pr_mask, num_classes):
    gt_mask = np.max(gt_mask * np.arange(1, num_classes), axis=-1)
    pr_mask = np.max(pr_mask * np.arange(1, num_classes), axis=-1)

    confusion_matrix = np.zeros((num_classes, num_classes))
    for row, col in np.ndindex(gt_mask.shape):
        gt_cls = gt_mask[row][col]
        pr_cls = pr_mask[row][col]
        confusion_matrix[gt_cls][pr_cls] += 1

    return confusion_matrix


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Evaluate a semantic segmentation model on KITTI dataset.')

    parser.add_argument('-c', '--model_cfg', required=True,
                        metavar='/path/to/model.cfg',
                        help='Path to model.cfg file')
    parser.add_argument('-w', '--weights', required=False,
                        default=None,
                        metavar='/path/to/weights.h5',
                        help='Path to weights.h5 file')
    parser.add_argument('-d', '--dataset', required=True,
                        metavar='/path/to/coco/',
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--tag', required=False,
                        default='simple',
                        metavar='<tag>',
                        help='Tag of the KITTI dataset (default=simple)')
    parser.add_argument('--subset', required=False,
                        default='val',
                        metavar="<subset>",
                        help='Either train or val')
    parser.add_argument('-t', '--threshold', required=False,
                        type=float,
                        default=0.5,
                        metavar='Threshold value for inference',
                        help='Must be between 0 and 1.')
    args = parser.parse_args()

    model = get_model_wrapper(load_config(args.model_cfg))
    model.load_weights(args.weights)

    dataset = KittiDataset()
    dataset.load_kitti(args.dataset, args.subset, args.tag)

    assert dataset.num_classes == model.config.NUM_CLASSES

    num_classes = dataset.num_classes

    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in tqdm(range(dataset.num_images)):
        img, gt_mask = load_image_gt(dataset, i, model.config.IMAGE_SHAPE)
        pr_mask = model.predict(img.astype(np.float32), args.threshold)
        confusion_matrix += compute_confusion_matrix(gt_mask.astype(np.int), pr_mask.astype(np.int), num_classes)

    # The measures below are implementation of standard measures
    # introduced in the following paper:
    # http://www.bmva.org/bmvc/2013/Papers/paper0032/paper0032.pdf

    G = np.sum(confusion_matrix, axis=1)
    P = np.sum(confusion_matrix, axis=0)

    # OP(overall pixel accuracy)
    # accuracy measures the proportion of correctly labelled pixels
    # limitation: bias in the presence of very imbalanced classes
    overall_pixel = np.trace(confusion_matrix) / np.sum(G)

    # PC(per class accuracy)
    # The proportion of correctly labelled pixels for
    # each class and then averages over the classes
    # Therefore, the background region absorbs all
    # false alarms without affecting the object class accuracies
    # limitation: a strong drawback for datasets with a large background class
    per_class = []
    for i in range(num_classes):
        per_class.append(confusion_matrix[i][i] / G[i])

    # JI(jaccard index)
    # Measures the intersection over the union of the labelled segments
    # for each class and reports the average. Thus takes into account
    # both the false alarms and the missed values for each class
    # limitation: it evaluates the amount of pixels correctly labelled, but not necessarily how
    # accurate the segmentation boundaries are
    jaccard_index = []
    for i in range(num_classes):
        jaccard_index.append(confusion_matrix[i][i] / (G[i] + P[i] - confusion_matrix[i][i]))

    # Normalized confusion matrix
    ncm = confusion_matrix / np.stack((G,) * num_classes, axis=1)

    class_names = ['Background']
    class_names += dataset.class_names

    print('<Class Indexing>')
    for class_id, class_name in enumerate(class_names):
        print('\t{}: {}'.format(class_id + 1, class_name))

    np.set_printoptions(precision=2, suppress=True)
    print('\n<Confusion Matrix>\n', ncm*100)
    np.set_printoptions()

    print('\n<Pixel Accuracy>')
    print('Overall: %.2f%%' % (overall_pixel.item() * 100))
    for class_name, op in zip(class_names, per_class):
        print('\t{}: %.2f%%'.format(class_name) % (op.item() * 100))

    print('\n<Jaccard Index>')
    print('Overall: %.2f%%' % (np.sum(jaccard_index).item() / num_classes * 100))
    for class_name, ji in zip(class_names, jaccard_index):
        print('\t{}: %.2f%%'.format(class_name) % (ji.item() * 100))
