import argparse
import matplotlib.pyplot as plt

from visual_tools import GuiInferenceViewer

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run Mask R-CNN detector.')

    parser.add_argument('-c', '--model_cfg', required=True,
                        metavar='/path/to/model.cfg',
                        help='Path to model.cfg file')
    parser.add_argument('-i', '--image_dir', required=True,
                        metavar='/path/to/directory',
                        help='Path to a directory containing images')
    parser.add_argument('-w', '--weights', required=False,
                        default=None,
                        metavar='/path/to/weights.h5',
                        help='Path to maskrcnn weights.h5 file')
    parser.add_argument('-t', '--threshold', required=False,
                        type=float,
                        default=0.5,
                        metavar='Threshold value for inference',
                        help='Must be between 0 and 1.')
    args = parser.parse_args()

    assert args.threshold >= 0.0 and args.threshold < 1.0, \
        'Invalid threshold value {} given'.format(args.threshold)

    viewer = GuiInferenceViewer(args.image_dir, args.model_cfg, args.weights, args.threshold)
    plt.show()
