[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/fcaf3d-fully-convolutional-anchor-free-3d/3d-object-detection-on-scannetv2)](https://paperswithcode.com/sota/3d-object-detection-on-scannetv2?p=fcaf3d-fully-convolutional-anchor-free-3d)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/fcaf3d-fully-convolutional-anchor-free-3d/3d-object-detection-on-sun-rgbd-val)](https://paperswithcode.com/sota/3d-object-detection-on-sun-rgbd-val?p=fcaf3d-fully-convolutional-anchor-free-3d)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/fcaf3d-fully-convolutional-anchor-free-3d/3d-object-detection-on-s3dis)](https://paperswithcode.com/sota/3d-object-detection-on-s3dis?p=fcaf3d-fully-convolutional-anchor-free-3d)

## FCAF3D: Fully Convolutional Anchor-Free 3D Object Detection

**News**:
 * :rocket: June, 2023. We add ScanNet-pretrained S3DIS model and log significantly pushing forward state-of-the-art.
 * :fire: February, 2023. Feel free to visit our new FCAF3D-based 3D instance segmentation [TD3D](https://github.com/samsunglabs/td3d) and real-time 3D object detection [TR3D](https://github.com/samsunglabs/tr3d).
 * :fire: August, 2022. FCAF3D is [now](https://github.com/open-mmlab/mmdetection3d/pull/1703) fully [supported](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/fcaf3d) in [mmdetection3d](https://github.com/open-mmlab/mmdetection3d).
 * :fire: July, 2022. Our paper is accepted at [ECCV 2022](https://www.ecva.net/papers.php).
 * :fire: March, 2022. We have updated the [preprint](https://arxiv.org/abs/2112.00322) adding more comparison with fully convolutional [GSDN](https://arxiv.org/abs/2006.12356) baseline.
  * :fire: December, 2021. FCAF3D is now state-of-the-art on [paperswithcode](https://paperswithcode.com/) on ScanNet, SUN RGB-D, and S3DIS.

This repository contains an implementation of FCAF3D, a 3D object detection method introduced in our paper:

> **FCAF3D: Fully Convolutional Anchor-Free 3D Object Detection**<br>
> [Danila Rukhovich](https://github.com/filaPro),
> [Anna Vorontsova](https://github.com/highrut),
> [Anton Konushin](https://scholar.google.com/citations?user=ZT_k-wMAAAAJ)
> <br>
> Samsung Research<br>
> https://arxiv.org/abs/2112.00322

### Installation
For convenience, we provide a [Dockerfile](docker/Dockerfile).

Alternatively, you can install all required packages manually. This implementation is based on [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) framework.
Please refer to the original installation guide [getting_started.md](docs/getting_started.md), replacing `open-mmlab/mmdetection3d` with `samsunglabs/fcaf3d`.
Also, [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) and [rotated_iou](https://github.com/lilanxiao/Rotated_IoU) should be installed with [these](https://github.com/samsunglabs/fcaf3d/blob/master/docker/Dockerfile#L22-L33) commands.

Most of the `FCAF3D`-related code locates in the following files: 
[detectors/single_stage_sparse.py](mmdet3d/models/detectors/single_stage_sparse.py),
[dense_heads/fcaf3d_neck_with_head.py](mmdet3d/models/dense_heads/fcaf3d_neck_with_head.py),
[backbones/me_resnet.py](mmdet3d/models/backbones/me_resnet.py).

### Getting Started

Please see [getting_started.md](docs/getting_started.md) for basic usage examples.
We follow the `mmdetection3d` data preparation protocol described in [scannet](data/scannet), [sunrgbd](data/sunrgbd), and [s3dis](data/s3dis).
The only difference is that we [do not sample](tools/data_converter/sunrgbd_data_utils.py#L143) 50,000 points from each point cloud in `SUN RGB-D`, using all points.

**Training**

To start training, run [dist_train](tools/dist_train.sh) with `FCAF3D` [configs](configs/fcaf3d):
```shell
bash tools/dist_train.sh configs/fcaf3d/fcaf3d_scannet-3d-18class.py 2
```

**Testing**

Test pre-trained model using [dist_test](tools/dist_test.sh) with `FCAF3D` [configs](configs/fcaf3d):
```shell
bash tools/dist_test.sh configs/fcaf3d/fcaf3d_scannet-3d-18class.py \
    work_dirs/fcaf3d_scannet-3d-18class/latest.pth 2 --eval mAP
```

**Visualization**

Visualizations can be created with [test](tools/test.py) script. 
For better visualizations, you may set `score_thr` in configs to `0.20`:
```shell
python tools/test.py configs/fcaf3d/fcaf3d_scannet-3d-18class.py \
    work_dirs/fcaf3d_scannet-3d-18class/latest.pth --eval mAP --show \
    --show-dir work_dirs/fcaf3d_scannet-3d-18class
```

### Models

The metrics are obtained in 5 training runs followed by 5 test runs. We report both the best and the average values (the latter are given in round brackets).

For `VoteNet` and `ImVoteNet`, we provide the configs and checkpoints with our Mobius angle parametrization.
For `ImVoxelNet`, please refer to the [imvoxelnet](https://github.com/saic-vul/imvoxelnet) repository as it is not currently supported in `mmdetection3d` for indoor datasets.
Inference speed (scenes per second) is measured on a single NVidia GTX1080Ti. Please, note that ScanNet-pretrained S3DIS model was actually trained in the original
[open-mmlab/mmdetection3d](https://github.com/open-mmlab/mmdetection3d/tree/main/configs/fcaf3d) codebase, so it can be inferenced only in their repo.

**FCAF3D**

| Dataset | mAP@0.25 | mAP@0.5 | Download |
|:-------:|:--------:|:-------:|:--------:|
| ScanNet | 71.5 (70.7) | 57.3 (56.0) | [model](https://github.com/samsunglabs/fcaf3d/releases/download/v1.0/20211007_144747.pth) &#124; [log](https://github.com/samsunglabs/fcaf3d/releases/download/v1.0/20211007_144747_fcaf3d_scannet.log) &#124; [config](configs/fcaf3d/fcaf3d_scannet-3d-18class.py) |
| SUN RGB-D | 64.2 (63.8) | 48.9 (48.2) | [model](https://github.com/samsunglabs/fcaf3d/releases/download/v1.0/20211007_144908.pth) &#124; [log](https://github.com/samsunglabs/fcaf3d/releases/download/v1.0/20211007_144908_fcaf3d_sunrgbd.log) &#124; [config](configs/fcaf3d/fcaf3d_sunrgbd-3d-10class.py) |
| S3DIS | 66.7 (64.9) | 45.9 (43.8) | [model](https://github.com/samsunglabs/fcaf3d/releases/download/v1.0/20211011_084059.pth) &#124; [log](https://github.com/samsunglabs/fcaf3d/releases/download/v1.0/20211011_084059_fcaf3d_s3dis.log) &#124; [config](configs/fcaf3d/fcaf3d_s3dis-3d-5class.py) |
| S3DIS <br> ScanNet-pretrained | 75.7 (74.1) | 58.2 (56.1) | [model](https://github.com/SamsungLabs/fcaf3d/releases/download/v1.0/20230601_131153_fcaf3d_scannet-pretrain_s3dis.pth) &#124; [log](https://github.com/SamsungLabs/fcaf3d/releases/download/v1.0/20230601_131153_fcaf3d_scannet-pretrain_s3dis.log) &#124; [config](https://github.com/SamsungLabs/fcaf3d/releases/download/v1.0/20230601_131153_fcaf3d_scannet-pretrain_s3dis.py) |


**Faster FCAF3D on ScanNet**

| Backbone | Voxel <br> size | mAP@0.25 | mAP@0.5 | Scenes <br> per sec. | Download |
|:--------:|:---------------:|:--------:|:-------:|:--------------------:|:--------:|
| HDResNet34 | 0.01 | 70.7 | 56.0 | 8.0 | see table above |
| HDResNet34:3 | 0.01 | 69.8 | 53.6 | 12.2 | [model](https://github.com/samsunglabs/fcaf3d/releases/download/v1.0/20211008_191702.pth) &#124; [log](https://github.com/samsunglabs/fcaf3d/releases/download/v1.0/20211008_191702_fcaf3d_3scales_scannet.log) &#124; [config](configs/fcaf3d/fcaf3d_3scales_scannet-3d-18class.py) |
| HDResNet34:2 | 0.02 | 63.1 | 46.8 | 31.5 | [model](https://github.com/samsunglabs/fcaf3d/releases/download/v1.0/20211008_151041.pth) &#124; [log](https://github.com/samsunglabs/fcaf3d/releases/download/v1.0/20211008_151041_fcaf3d_2scales_scannet.log) &#124; [config](configs/fcaf3d/fcaf3d_2scales_scannet-3d-18class.py) |

**VoteNet on SUN RGB-D**

| Source | mAP@0.25 | mAP@0.5 | Download |
|:------:|:--------:|:-------:|:--------:|
| mmdetection3d | 59.1 | 35.8| [instruction](configs/votenet) |
| ours | 61.1 (60.5) | 40.4 (39.5) | [model](https://github.com/samsunglabs/fcaf3d/releases/download/v1.0/20211016_132950.pth) &#124; [log](https://github.com/samsunglabs/fcaf3d/releases/download/v1.0/20211016_132950_votenet_sunrgbd.log) &#124; [config](configs/votenet/votenet-v2_16x8_sunrgbd-3d-10class.py) |

**ImVoteNet on SUN RGB-D**

| Source | mAP@0.25 | mAP@0.5 | Download |
|:------:|:--------:|:-------:|:--------:|
| mmdetection3d | 64.0 | 37.8 | [instruction](configs/imvotenet) |
| ours | 64.6 (64.1) | 40.8 (39.8) | [model](https://github.com/samsunglabs/fcaf3d/releases/download/v1.0/20211009_131500.pth) &#124; [log](https://github.com/samsunglabs/fcaf3d/releases/download/v1.0/20211009_131500_imvotenet_sunrgbd.log) &#124; [config](configs/imvotenet/imvotenet-v2_stage2_16x8_sunrgbd-3d-10class.py) |

**Comparison with state-of-the-art on ScanNet**

<p align="center"><img src="./resources/scannet_map_fps.png" alt="drawing" width="50%"/></p>

### Example Detections

<p align="center"><img src="./resources/github.png" alt="drawing" width="90%"/></p>

### Citation

If you find this work useful for your research, please cite our paper:
```
@inproceedings{rukhovich2022fcaf3d,
  title={FCAF3D: fully convolutional anchor-free 3D object detection},
  author={Rukhovich, Danila and Vorontsova, Anna and Konushin, Anton},
  booktitle={European Conference on Computer Vision},
  pages={477--493},
  year={2022},
  organization={Springer}
}
```
