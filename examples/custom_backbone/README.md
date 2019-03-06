# Custom Backbone
There are a few steps to use your own backbone CNN.
1. Implement a function that returns your keras.Model object as in [matterport_resnet/models.py](https://github.com/nearthlab/image-segmentation/blob/master/examples/custom_backbone/matterport_resnet/models.py)
2. Update the Classifier, for example, as in [train.py](https://github.com/nearthlab/image-segmentation/blob/90a2a96660fb2e564de0d4f9d4593d5ed326bfe2/examples/custom_backbone/train.py#L14). You can give your backbone a new name or override the existing backbone name (which is what this example does).

To run example inference code on image files in images folder, download MaskRCNN_coco.h5 from [releases page](https://github.com/nearthlab/image-segmentation/releases) in this folder. And run:
```bash
  cd /path/to/image-segmentation/examples/custom_backbone
  python infer_gui.py
  # or python infer.py to save the results in images/results folder, which will be autamatically created
```

To train your model with this example custom backbone, run train.py:
```bash
  cd /path/to/image-segmentation/examples/custom_backbone
  python train.py -d /path/to/coco
```

To evaluate the model, run evaluate.py:
```bash
  cd /path/to/image-segmentation/examples/custom_backbone
  python evaluate.py -d /path/to/coco
```