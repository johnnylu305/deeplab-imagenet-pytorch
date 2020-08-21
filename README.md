# An unofficial Deeplab V2 with the pre-train weight of ImageNet
This repository and codes are largely based on and modified from [deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch). I highly recommend visiters to visit the GitHub.

Performance on Pascal VOC12


| set      | CRF      | mIoU     |
| :---:    | :---:    |  :---:   |
| val    |O         | 78.8%   |
| test      |O         | 79.1%   |

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

It can reach 77% mIoU and 78% mIoU after CRF on Pascal VOC12 val.

## Further details

[kazuto1011/deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch)

## References

1. L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, A. L. Yuille. DeepLab: Semantic Image
Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs. *IEEE TPAMI*,
2018.<br>
[Project](http://liangchiehchen.com/projects/DeepLab.html) /
[Code](https://bitbucket.org/aquariusjay/deeplab-public-ver2) / [arXiv
paper](https://arxiv.org/abs/1606.00915)

2. H. Caesar, J. Uijlings, V. Ferrari. COCO-Stuff: Thing and Stuff Classes in Context. In *CVPR*, 2018.<br>
[Project](https://github.com/nightrome/cocostuff) / [arXiv paper](https://arxiv.org/abs/1612.03716)

1. M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, A. Zisserman. The PASCAL Visual Object
Classes (VOC) Challenge. *IJCV*, 2010.<br>
[Project](http://host.robots.ox.ac.uk/pascal/VOC) /
[Paper](http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham10.pdf)
