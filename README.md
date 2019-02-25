# image-segmentation

This repository includes:
  * A re-implementation of [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN) with **multiple backbone support** using the implementations of various backbone models in [qubvel/classification_models](https://github.com/qubvel/classification_models). (See [here](https://github.com/qubvel/classification_models#architectures) for available backbone architectures)
  * **Unified training, inference and evaluation** codes for Mask R-CNN and some semantic segmentation models (from [qubvel/segmentation_models](https://github.com/qubvel/segmentation_models)), for which you can **easily modify various parameters** with simple configuration files.
```
  [Available segmentation models]
  Instance: maskrcnn
  Semantic: fpn, linknet, pspnet, unet
```

# Installation

**How to set up a virtual environment and install on it**
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
  cd /path/to/works/image-segmentation
  source activate
  # this is equivalent to: 
  # source ../venv/bin/activate && export PYTHONPATH=`pwd`/image-segmentation
  # i.e. activating the virtual environment and add the image-segmentation folder to the PYTHONPATH
```
  
**How to install without a virtual environment**
  Note that working on a virtual environment is highly recommended. But if you insist on not using it, you can still do so:
```bash
  git clone https://github.com/nearthlab/image-segmentation
  cd image-segmentation 
  pip install -r requirements.txt
```

**Requirements**

    1. Python 3.5+
    2. segmentation-models==0.2.0
    3. keras>=2.2.0
    4. keras-applications>=1.0.7 
    5. tensorflow(-gpu)==1.10.0 (tested)




