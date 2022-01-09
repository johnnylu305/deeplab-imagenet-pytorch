# An unofficial Deeplab V2 with the pre-train weight of ImageNet
This repository and codes are largely based on and modified from [deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch). I highly recommend visiters to visit the GitHub.

## Performance
- [Performance on Pascal VOC12 (Image size: 381, Batch size: 26)](https://drive.google.com/drive/folders/1aSGciJqmecgUG6R9427YMjUJl2JZzTae?usp=sharing)

| set      | CRF      | mIoU     |
| :---:    | :---:    |  :---:   |
| val    |O         | 78.8%   |
| test      |O         | 79.1%   |

- [Performance on Pascal VOC12 (Image size: 321, Batch size: 12)](https://drive.google.com/drive/folders/1aU0MPZYuDypHRR56pyXLZbuqzhin2pD0?usp=sharing)

| set      | CRF      | mIoU     |
| :---:    | :---:    |  :---:   |
| val    |O         | 77.02%   |

## My Environment
- Operating System:
  - Ubuntu 16.04.5
- CUDA:
  - CUDA V10.0.130 
- Nvidia driver:
  - 418.87.01
- Python:
  - python 3.6.8
- Python package:
  - tqdm, opencv-python, pydensecrf,...
- Pytorch:
  - pytorch-gpu 1.4.0

<!-- - Memory- 128GB -->
<!-- - GPU:- Tesla v100 30G * 2 -->

## Downloading the VOC12 dataset
[Visual Object Classes Challenge 2012 (VOC2012)](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
## Download the VOC12 augmentation dataset
- [Semantic Boundaries Dataset and Benchmark](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0)


## For training

```
python main.py train --config-path configs/voc12ImageNet.yaml
```

## For testing

```
python main.py test --config-path configs/voc12Test.yaml
```

## For crf

```
python main.py crf --config-path configs/voc12Test.yaml --n-jobs 8
```

## OOM
- Please try small batch size and image size
- Disable evaluate during training

## Further details

[kazuto1011/deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch)

## References

1. [kazuto1011/deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch)

2. L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, A. L. Yuille. DeepLab: Semantic Image
Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs. *IEEE TPAMI*,
2018.<br>
[Project](http://liangchiehchen.com/projects/DeepLab.html) /
[Code](https://bitbucket.org/aquariusjay/deeplab-public-ver2) / [arXiv
paper](https://arxiv.org/abs/1606.00915)

3. H. Caesar, J. Uijlings, V. Ferrari. COCO-Stuff: Thing and Stuff Classes in Context. In *CVPR*, 2018.<br>
[Project](https://github.com/nightrome/cocostuff) / [arXiv paper](https://arxiv.org/abs/1612.03716)

4. M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, A. Zisserman. The PASCAL Visual Object
Classes (VOC) Challenge. *IJCV*, 2010.<br>
[Project](http://host.robots.ox.ac.uk/pascal/VOC) /
[Paper](http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham10.pdf)
