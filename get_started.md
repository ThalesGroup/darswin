# DarSwin Transformer for Image Classification

This folder contains the implementation of the DarSwin Transformer for image classification.

## Usage

### Install

We recommend using the pytorch docker `nvcr>=21.05` by
nvidia: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch.

- Clone this repo:

```bash
git clone https://github.com/ThalesGroup/darswin.git
cd darswin
```

- Create a conda virtual environment and activate it:

```bash
conda create -n darswin python=3.7 -y
conda activate darswin
```

- Install `CUDA>=10.2` with `cudnn>=7` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `PyTorch>=1.8.0` and `torchvision>=0.9.0` with `CUDA>=10.2`:

```bash
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch
```

- Install `timm==0.4.12`:

```bash
pip install timm==0.4.12
```

- Install other requirements:

```bash
pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8 imageio
```



### Data preparation

We use standard ImageNet2010 dataset, you can download it from http://image-net.org/. We provide the following two ways to
load data:

- For standard folder dataset, move validation images to labeled sub-folders. The file structure should look like:
  ```bash
  $ tree data
  imagenet
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...
 
  ```
- We use 200 classes from ImageNet21k in DarSwin/data/imagenet_2010/classes.pkl
    - `DarSwin/data/imagenet_2010/train/train.pkl`, `DarSwin/data/imagenet_2010/val/val.pkl`: which store the names of images for train and validate splits.
    - `DarSwin/data/imagenet_2010/test_cls/test_cls.pkl`: which store the names of images for train and validate splits.
    - `DarSwin/data/imagenet_2010/test_cls/test_1.pkl` : which stores images with distortion parameter $\xi \in [0,0.05]$, similary for `test_2.pkl`, `test_3.pkl`, `test_4.pkl` stores images with distortion parameter $\xi \in [0.2, 0.35]$, $\xi \in [0.5, 0.7]$, $\xi \in [0.85, 0.93]$
- The `DarSwin/data/Distorted_imagenet.py` is used to load the images and synthtically distort using spherical distortion.

### Evaluation

To evaluate a pre-trained `DarSwin Transformer` on ImageNet val, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345 main.py --eval \
--cfg <config-file> --resume <checkpoint> --task <level of distortion> --data-path <imagenet-path> 
```

For example, to evaluate the `DarSwin-angular positional encoding on low level distortion` with a single GPU:

```bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval \
--cfg configs/swin/one_distortion_swin_small_patch2_window4_64_gp2_angular.yaml --resume one_distortion_swin_small_patch2_window4_64_gp2_angular.yaml.pth --task <level of distortion> --data-path <imagenet-path>
```

- Level of distortion is defined by :
  -  test_1 : Very Low distorted
  -  test_2 : Low distorted
  -  test_3 : Medium distorted
  -  test_4 : High distorted

### eval.sh and eval_xi.sh #######
```bash
./eval.sh : evaluate on different levels of distortion by definining the --task [test_1, test_2, test_3, test_4] depending on level of ditrotion for testing 
./eval_xi.sh : evaluate on all values $\xi \in [0, 1]$
```
### Training from scratch on ImageNet-1K

### checkpoints 
- Different levels of distoriton has different checkpoint file, they can be found at TBD
  -  gp1 : Very Low distorted
  -  gp2 : Low distorted
  -  gp3 : Medium distorted
  -  gp4 : High distorted


To train a `DarSwin Transformer` on ImageNet from scratch, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345  main.py \ 
--cfg <config-file> --data-path <imagenet-path> [--batch-size <batch-size-per-gpu> --output <output-directory> --tag <job-tag>]
```

### Throughput

To measure the throughput, run:

```bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py \
--cfg <config-file> --data-path <imagenet-path> --batch-size 64 --throughput --disable_amp
```
