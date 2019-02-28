import os
import argparse
from shutil import copyfile

from models import get_model_wrapper
from keras_model_wrapper import Trainer
from config import load_config

############################################################
#  Training
############################################################


if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on Unreal COCO.')
    parser.add_argument('-d', '--dataset', required=True,
                        metavar='/path/to/coco/',
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--tag', required=False,
                        default='2017',
                        metavar='<tag>',
                        help='Tag of the MS-COCO dataset (default=2017)')
    parser.add_argument('-w', '--weights', required=False,
                        default=None,
                        metavar='/path/to/weights.h5',
                        help='Path to maskrcnn weights.h5 file')
    parser.add_argument('-m', '--model_cfg', required=True,
                        metavar='/path/to/model.cfg',
                        help='Path to model.cfg file')
    parser.add_argument('-t', '--train_cfg', required=True,
                        nargs='+',
                        metavar='path/to/train.cfg',
                        help='Path to train.cfg file(s) [More than one can be provided]')
    parser.add_argument('-s', '--workspace', required=False,
                        default='logs',
                        metavar='/path/to/workspace/',
                        help='Workspace is a parent directory containing log directories (default=logs)')
    args = parser.parse_args()

    print('Model weights: ', args.weights)
    print('Dataset: ', args.dataset)
    print('Tag: ', args.tag)
    print('Workspace: ', args.workspace)

    # Model Configuration
    model_config = load_config(args.model_cfg)

    # Create model
    print('Building model...')
    model_wrapper = get_model_wrapper(model_config)

    # Training Configurations
    train_cfgs = []
    for path in args.train_cfg:
        train_cfgs.append(load_config(path))

    for stage, train_config in enumerate(train_cfgs):
        # Create trainer
        trainer = Trainer(model_wrapper=model_wrapper,train_config=train_config,
                          workspace=args.workspace, stage=stage)

        # Load the weights file designated by the command line argument
        # only on the first stage and continue training on the current
        # model's weights in the later stages
        if stage == 0:
            if args.weights is not None:
                if args.weights.lower() == 'last':
                    # Find the last trained weights
                    model_path = trainer.find_last()
                    trainer.set_log_dir(model_path, inherit_epoch=True)
                elif args.weights.lower() == 'best':
                    model_path = trainer.find_last(by_val_loss=True)
                    trainer.set_log_dir(model_path, inherit_epoch=True)
                else:
                    model_path = args.weights
                    trainer.set_log_dir(model_path)

                # Load weights
                print('Loading weights', model_path)
                model_wrapper.load_weights(model_path)

            else:
                trainer.set_log_dir()
                print('No weights provided. Will use randomly initialized weights')
        else:
            best_weight_path = trainer.find_last(by_val_loss=True)
            print('Loading weights', best_weight_path)
            model_wrapper.load_weights(best_weight_path)
            trainer.set_log_dir(best_weight_path)

        trainer.train(dataset_dir=args.dataset, tag=args.tag)

        if stage == len(train_cfgs) - 1:
            best_weight_path = trainer.find_last(by_val_loss=True)
            print('The best weights in the final stage is {}'.format(best_weight_path))
            copy_dst = os.path.normpath(os.path.join(trainer.log_dir, '..', 'best_model.h5'))
            copyfile(best_weight_path, copy_dst)
            print('The best weights file is copied to {}'.format(copy_dst))
