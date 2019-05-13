# This file contains (modified) parts of the codes from the following repository:
# https://github.com/matterport/Mask_RCNN
#
# Mask R-CNN
#
# The MIT License (MIT)
#
# Copyright (c) 2017 Matterport, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os
import re
import json
import datetime
import multiprocessing
import tensorflow as tf

from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.optimizers import Adagrad
from keras.optimizers import Adadelta
from keras.optimizers import Adam
from keras.optimizers import Adamax
from keras.optimizers import Nadam

from keras.regularizers import l2

from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TerminateOnNaN
from keras.callbacks import CSVLogger


# See https://keras.io/optimizers/ for details
def get_optimizer(config):
    if config.OPTIMIZER == 'SGD':
        return SGD(lr=config.LEARNING_RATE, momentum=config.LEARNING_MOMENTUM, clipnorm=config.GRADIENT_CLIP_NORM, nesterov=config.NESTEROV)
    elif config.OPTIMIZER == 'RMSprop':
        return RMSprop(lr=config.LEARNING_RATE, clipnorm=config.GRADIENT_CLIP_NORM)
    elif config.OPTIMIZER == 'Adagrad':
        return Adagrad(lr=config.LEARNING_RATE, clipnorm=config.GRADIENT_CLIP_NORM)
    elif config.OPTIMIZER == 'Adadelta':
        return Adadelta(lr=config.LEARNING_RATE, clipnorm=config.GRADIENT_CLIP_NORM)
    elif config.OPTIMIZER == 'Adam':
        return Adam(lr=config.LEARNING_RATE, clipnorm=config.GRADIENT_CLIP_NORM, amsgrad=config.AMSGRAD)
    elif config.OPTIMIZER == 'Adamax':
        return Adamax(lr=config.LEARNING_RATE, clipnorm=config.GRADIENT_CLIP_NORM)
    elif config.OPTIMIZER == 'Nadam':
        return Nadam(lr=config.LEARNING_RATE, clipnorm=config.GRADIENT_CLIP_NORM)
    else:
        raise Exception('Unrecognized optimizer: {}'.format(config.OPTIMIZER))


class Trainer:

    def __init__(self, model_wrapper, train_config, workspace, stage=0, check_sanity=False):
        assert model_wrapper.config.MODE == 'training', 'Create model in training mode.'
        self.branched_train_config = train_config
        self.train_config = train_config.flatten()
        self.workspace = workspace
        if not os.path.exists(self.workspace):
            os.makedirs(self.workspace)
        self.branched_model_config = model_wrapper.branched_config
        self.model_config = model_wrapper.config
        self.keras_model = model_wrapper.model
        self.backbone_layer_names = model_wrapper.backbone_layer_names
        self.log_dir = None
        self.name = self.keras_model.inner_model.name if self.keras_model.__class__.__name__ == 'ParallelModel' \
            else self.keras_model.name
        self.stage = stage
        self.check_sanity = check_sanity


    def compile(self):
        '''Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        '''

        # Optimizer object
        optimizer = get_optimizer(self.train_config)

        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = self.train_config.LOSS_WEIGHTS.keys()
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            loss = (
                    tf.reduce_mean(layer.output, keepdims=True)
                    * self.train_config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.add_loss(loss)

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [
            l2(self.train_config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
            for w in self.keras_model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name]
        assert len(reg_losses) > 0, 'No trainable weights found. Check the TRAINABLE_LAYERS parameter in train.cfg\n{}'.format(self.branched_train_config)
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.keras_model.compile(
            optimizer=optimizer,
            loss=[None] * len(self.keras_model.outputs))

        # Add metrics for losses
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            loss = (
                    tf.reduce_mean(layer.output, keepdims=True)
                    * self.train_config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.metrics_tensors.append(loss)


    def get_data_generator(self, dataset_dir, subset, tag):

        if self.model_config.MODEL == 'maskrcnn':
            from data_generators.coco import data_generator, CocoDataset

            dataset = CocoDataset()
            print('Loading COCO Dataset: {} (subset: {} / version tag: {})'.format(dataset_dir, subset, tag))
            dataset.load_coco(dataset_dir, subset, tag)
            dataset.prepare()
        else:
            from data_generators.kitti import data_generator, KittiDataset

            dataset = KittiDataset()
            print('Loading KITTI Dataset: {} (subset: {} / version tag: {})'.format(dataset_dir, subset, tag))
            dataset.load_kitti(dataset_dir, subset, tag)
            if self.stage == 0 and self.check_sanity:
                print('Checking sanity of the dataset...')
                dataset.check_sanity()

        print('num_images: {} / num_classes: {}'.format(dataset.num_images, dataset.num_classes))
        assert dataset.num_classes == self.model_config.NUM_CLASSES, 'NUM_CLASSES in model and dataset mismatched.'

        if self.stage == 0:
            fp = open(os.path.join(self.log_dir, '..', 'class_names.json'), 'w')
            json.dump(dataset.class_names, fp)
            fp.close()

        return data_generator(dataset, self.model_config, shuffle=True, batch_size=self.model_config.BATCH_SIZE)


    def find_last(self, by_val_loss=False):
        '''Finds the last checkpoint file of the last trained model in the
                        model directory.
                        Returns:
                            The path of the last checkpoint file
                        '''

        def find_last_dir_name_starting_with(parent_dir, prefix):
            dir_names = next(os.walk(parent_dir))[1]
            dir_names = filter(lambda f: f.startswith(prefix), dir_names)
            dir_names = sorted(dir_names)
            if not dir_names:
                import errno
                raise FileNotFoundError(
                    errno.ENOENT,
                    'Could not find model directory under {}'.format(self.workspace))
            return os.path.join(parent_dir, dir_names[-1])

        # Find the latest log directory
        latest_log_dir = find_last_dir_name_starting_with(self.workspace, self.name)

        # Find last stage directory
        last_stage_dir = find_last_dir_name_starting_with(latest_log_dir, 'stage')

        match = re.findall('stage(\d)', last_stage_dir)
        if match:
            self.stage = int(match[0])

        # Find the last checkpoint
        checkpoints = next(os.walk(last_stage_dir))[2]
        checkpoints = sorted([x for x in checkpoints if x.endswith('.h5')])

        if not checkpoints:
            import errno
            raise FileNotFoundError(
                errno.ENOENT, 'Could not find weight files in {}'.format(last_stage_dir))

        if by_val_loss:
            checkpoints = sorted(checkpoints,
                                 key=lambda x: float(x[:-3].split(sep='_')[-1]),
                                 reverse=True)

        checkpoint = os.path.join(last_stage_dir, checkpoints[-1])

        return checkpoint


    def set_trainable(self, layer_regex, train_backbone, train_bn=False, keras_model=None, indent=0, verbose=True):
        '''Sets model layers as trainable if their names match
        the given regular expression.
        '''
        # Print message on the first call (but not on recursive calls)
        if verbose and keras_model is None:
            print('Selecting layers to train')

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, 'inner_model') \
            else keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print('In model: ', layer.name)
                self.set_trainable(
                    layer_regex, train_backbone=train_backbone, train_bn=train_bn, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue

            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name)) \
                        or (train_backbone and layer.name in self.backbone_layer_names)

            # Make BatchNormalization layers trainable only when train_bn is True
            if 'BatchNorm' in layer.__class__.__name__ or \
                    (layer.__class__.__name__ == 'TimeDistributed' and 'BatchNorm' in layer.layer.__class__.__name__):
                trainable = trainable and train_bn

            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable

            # Print layer status
            if verbose:
                msg = '{}{:20}   ({:18}):     {}'.format(' ' * indent, layer.name,
                                                      layer.__class__.__name__ if layer.__class__.__name__ != 'TimeDistributed'
                                                         else layer.layer.__class__.__name__,
                                                      'Trainable' if trainable else 'Frozen')
                print(msg)
                fp = open(os.path.join(self.log_dir, 'layer_status.log'), 'a')
                fp.write(msg + '\n')
                fp.close()


    def set_log_dir(self, model_path=None, inherit_epoch=False):
        '''Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        '''
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # \path\to\logs\coco20171029T2315\mask_rcnn_coco_0001.h5 (Windows)
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5 (Linux)

            regex = r'.*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]%s[/\\][\w-]+(\d{4})' % 'stage{}'.format(self.stage)
            m = re.match(regex, model_path)

            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))

                if inherit_epoch:
                    # Epoch number in file is 1-based, and in Keras code it's 0-based.
                    # So, adjust for that then increment by one to start from the next epoch
                    self.epoch = int(m.group(6))
                    print('Re-starting from epoch %d' % self.epoch)
                    self.stage -= 1 if self.stage > 0 else 0

        # Directory for training logs
        if self.log_dir is None or model_path is not None:
            self.log_dir = os.path.join(self.workspace, '{}D{:%Y%m%dT%H%M}/stage{}'.format(
                self.name, now, self.stage + 1))
        # Create log_dir if it does not exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, '{}_{}_*epoch*_*val_loss*.h5'.format(
            self.branched_train_config.NAME.lower(), self.name))
        self.checkpoint_path = self.checkpoint_path.replace(
            '*epoch*', '{epoch:04d}').replace(
            '*val_loss*', '{val_loss:.4f}'
        )


    def make_inference_config(self):
        infer_config = self.branched_model_config
        infer_config.BACKBONE.BACKBONE_WEIGHTS = 'None'
        infer_config.INPUT.GPU_IDS = [0]
        infer_config.INPUT.IMAGES_PER_GPU = 1
        infer_config.MODEL.MODE = 'inference'
        return infer_config


    def train(self, dataset_dir, tag, custom_callbacks=None):
        '''
        Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        custom_callbacks: Optional. Add custom callbacks to be called
            with the keras fit_generator method. Must be list of type keras.callbacks.
        '''

        self.branched_train_config.save(os.path.join(self.log_dir, 'train_stage{}.cfg'.format(self.stage + 1)))
        if self.stage == 0:
            self.branched_model_config.save(os.path.join(self.log_dir, '..', 'model.cfg'))
            infer_config = self.make_inference_config()
            infer_config.save(os.path.join(self.log_dir, '..', 'infer.cfg'))

            summary_fp = open(os.path.join(self.log_dir, '..', 'model_summary.log'), 'w')
            self.keras_model.summary(print_fn=lambda x: summary_fp.write(x + '\n'))
            summary_fp.close()
        
        assert self.log_dir is not None, 'You should call Trainer::set_log_dir() before start training'

        print('Logging directory:', self.log_dir)

        if self.train_config.TRAINABLE_LAYERS == 'all':
            layer_regex = r'.*'
            train_backbone = True
        else:
            layer_regex = '|'.join([r'{}.*'.format(x) for x in self.train_config.TRAINABLE_LAYERS if x != 'backbone'])
            train_backbone = 'backbone' in self.train_config.TRAINABLE_LAYERS

        # Data generators
        train_generator = self.get_data_generator(dataset_dir, 'train', tag)
        val_generator = self.get_data_generator(dataset_dir, 'val', tag)

        if hasattr(train_generator, 'class_names') and hasattr(val_generator, 'class_names'):
            assert train_generator.class_names == val_generator.class_names

        # Callbacks
        callbacks = [
            TensorBoard(log_dir=self.log_dir, histogram_freq=0, write_graph=(self.stage==0)),
            ModelCheckpoint(self.checkpoint_path, verbose=0, save_weights_only=True),
            TerminateOnNaN(),
            CSVLogger(os.path.join(self.log_dir, 'loss_stage{}.csv'.format(self.stage + 1)), separator=',', append=True)
        ]

        # Add custom callbacks to the list
        if custom_callbacks:
            callbacks += custom_callbacks

        learning_rate = self.train_config.LEARNING_RATE
        epochs = self.train_config.EPOCHS

        # Train
        print('\nStarting at epoch {}. LR={}\n'.format(self.epoch, learning_rate))
        print('Checkpoint Path: {}'.format(self.checkpoint_path))
        self.set_trainable(layer_regex, train_backbone=train_backbone, train_bn=self.train_config.TRAIN_BN)
        self.compile()

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name is 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        print('===================Training Stage {}==================='.format(self.stage + 1))
        self.branched_train_config.display()

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.train_config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.train_config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=True,
        )
        print('===================Stage {} Finished===================\n\n'.format(self.stage + 1))
