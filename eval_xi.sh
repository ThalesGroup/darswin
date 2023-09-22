#!/bin/bash 
CONFIG=one_distortion_swin_small_patch2_window4_64_gp2_angular.yaml

export WANDB_MODE="disabled"


for i in $(seq 0.0 0.1 0.9)
do 
    echo $i
    python -m torch.distributed.launch --nproc_per_node=1 --master_port 12346 main.py --eval \
    --resume /home-local2/akath.extra.nobkp/checkpoints_spherical_pe/gp2_ra/ckpt_epoch_325.pth \
    --cfg configs/swin/$CONFIG                                              \
    --data-path /home-local2/akath.extra.nobkp/imagenet_2010     \
    --batch-size 128 \
    --xi $i \
    --task test \
    --xi_bool True  \ 
   
done    


# gp1 : ckpt_epoch_240.pth
# gp2 : ckpt_epoch_235.pth

# gp1_val : ckpt_epoch_255.pth
# gp2_val : ckpt_epoch_255.pth