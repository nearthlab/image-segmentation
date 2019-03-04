# image-segmentation

This repository includes:
  * A re-implementation of [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN) with **multiple backbone support** using the implementations of various backbone models in [qubvel/classification_models](https://github.com/qubvel/classification_models). (See [here](https://github.com/qubvel/classification_models#architectures) for available backbone architectures)
  * **Unified training, inference and evaluation** codes for Mask R-CNN and some semantic segmentation models (from [qubvel/segmentation_models](https://github.com/qubvel/segmentation_models)), for which you can **easily modify various parameters** with simple configuration files.
  * Coco dataset viewer
```
  [Available segmentation models]
  Instance: maskrcnn
  Semantic: fpn, linknet, pspnet, unet
```

# Installation

**i. How to set up a virtual environment and install on it**<br/>
```bash
  sudo apt-get install virtualenv
  mkdir works
  cd works
  virtualenv -p python3 venv
  git clone https://github.com/nearthlab/image-segmentation
  cd image-segmentation
  source activate 
  pip install -r requirements.txt
```
  
  You should run the following commands every time you open a new terminal in order to run any of python files
```bash
  cd /path/to/image-segmentation
  source activate
  # this is equivalent to: 
  # source ../venv/bin/activate && export PYTHONPATH=`pwd`/image-segmentation
  # i.e. activating the virtual environment and add the image-segmentation folder to the PYTHONPATH
```
  
**ii. How to install without a virtual environment**<br/>
  Note that working on a virtual environment is highly recommended. But if you insist on not using it, you can still do so:
```bash
  git clone https://github.com/nearthlab/image-segmentation
  cd image-segmentation
  pip install --user -r requirements.txt
```

**Requirements**<br/>

    1. Python 3.5+
    2. segmentation-models==0.2.0
    3. keras>=2.2.0
    4. keras-applications>=1.0.7 
    5. tensorflow(-gpu)==1.10.0 (tested)

# How to train your own model

  i. Download the modified KITTI dataset from [release page](https://github.com/nearthlab/image-segmentation/releases) and place it under datasets folder. [Note that the KITTI dataset is public and available [online](http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015). I simply splitted the dataset into training and validation dataset and simplified the labels.]

  ii. Choose your model and copy corresponding cfg files from examples/configs. For example, if you want to train a Unet model with resnet18 backbone,
```bash
  cd /path/to/image-segmentation
  mkdir unet_resnet18
  cp examples/configs/unet/*.cfg unet_resnet18
```

  iii. [Optional] Tune some model and training parameters in the config files that you have just copied. Read the comments in the example config files for what each parameter does.
[Note that you have to declare a variable in .cfg file in the format
```{type}-{VARIABLE_NAME} = {value}```]

  iv. Run the training script
```bash
  cd cd /path/to/image-segmentation
  python python train.py -s unet_resnet18 -d datasets/KITTI -m unet_resnet18/unet.cfg -t unet_resnet18/train_unet_decoder.cfg unet_resnet18/train_unet_all.cfg
```
  This script will train the unet model in two stages with training information in unet_resnet18/train_unet_decoder.cfg followed by unet_resnet18/train_unet_all.cfg. The idea is:
  we first train the decoder part only while freezing the backbone with imagenet-pretrained weights loaded,
  and then fine tune the entire model in the second stage. You can provide as many training cfg files as you wish, dividing training into multiple stages.
  
# 

