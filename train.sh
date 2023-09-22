#!/bin/bash 
export WANDB_MODE="disabled"


python -m torch.distributed.launch \
--nproc_per_node 1 \
--master_port 12346  main.py \
--cfg configs/swin/one_distortion_swin_small_patch2_window4_64_gp2_angular.yaml \
--resume /home-local2/akath.extra.nobkp/checkpoints_spherical_pe/gp2_ra/ckpt_epoch_325.pth \
--data-path /home-local2/akath.extra.nobkp/imagenet_2010/ \
--batch-size 8
 

 # ICCV (gp1) gp1_swin : 140, 130, 
  # ICCV (gp1_ra) gp1_pe : 145, 145,  