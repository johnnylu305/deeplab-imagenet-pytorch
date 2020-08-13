# Deeplab V2 with the pre-train weight of ImageNet.
This code is largely based on and modified from [deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch). I highly recommend visiters to visit the GitHub.

## Downloading the VOC12 dataset
[Visual Object Classes Challenge 2012 (VOC2012)](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)


## For training

```
python main.py train --config-path configs/voc12ImageNet.yaml
```

## For finetune
### This mode is still in testing stage.
```
python main.py finetune --config-path configs/voc12Finetune.yaml
```

## For testing

```
python main.py test --config-path configs/voc12Test.yaml
```

## For crf

```
python main.py crf --config-path configs/voc12Test.yaml --n-jobs 8
```

## Performance

It can reach x% mIoU and x% mIoU after CRF on Pascal VOC12 val.

## Further details

[kazuto1011/deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch)
