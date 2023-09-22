CONFIG=one_distortion_swin_small_patch2_window4_64_gp1_ra.yaml

export WANDB_MODE="disabled"
                                    
python -m torch.distributed.launch --nproc_per_node=4 --master_port 12346 main.py --eval    \
--resume /home-local2/akath.extra.nobkp/checkpoints_spherical_pe/gp2_ra/ckpt_epoch_325.pth \
--cfg configs/swin/$CONFIG                                              \
--data-path /home-local2/akath.extra.nobkp/imagenet_2010     \
--task test_1 \
--batch-size 64


#255, 250, 245, 250
