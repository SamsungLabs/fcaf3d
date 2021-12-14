FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6

# Install MMCV, MMDetection and MMSegmentation
RUN pip install mmcv-full==1.3.8 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.1/index.html
RUN pip install mmdet==2.14.0
RUN pip install mmsegmentation==0.14.1

# Install MMDetection3D
RUN conda clean --all
RUN git clone https://github.com/samsunglabs/fcaf3d.git /mmdetection3d
WORKDIR /mmdetection3d
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .

# Uninstall pycocotools installed by nuscenes-devkit and reinstall mmpycocotools
RUN pip uninstall pycocotools --no-cache-dir -y
RUN pip install mmpycocotools --no-cache-dir --force --no-deps

# Install other packages
RUN pip install open3d==0.12.0

# Install Minkowski Engine
RUN apt-get install -y python3-dev libopenblas-dev
RUN pip install ninja
RUN pip install -U git+https://github.com/NVIDIA/MinkowskiEngine --install-option="--blas=openblas" -v --no-deps

# Install differentiable IoU
RUN git clone https://github.com/lilanxiao/Rotated_IoU /rotated_iou
RUN cp -r /rotated_iou/cuda_op /mmdetection3d/mmdet3d/ops/rotated_iou
WORKDIR /mmdetection3d/mmdet3d/ops/rotated_iou/cuda_op
RUN python setup.py install
