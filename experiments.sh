#!/bin/sh
python train_fsd50k_chunks.py --cfg_file ./cfgs/vgglike_chunks.cfg -e ../experiments/fsd50k-pytorch/vgglike_adam_256_bg_aug --resume_from /home/sarthak/Workspace/experiments/fsd50k-pytorch/vgglike_adam_256_bg_aug/ckpts/epoch=09_train_mAP=0.393_val_mAP=0.376.ckpt
python train_fsd50k_chunks.py --cfg_file ./cfgs/resnet18_chunks.cfg -e ../experiments/fsd50k-pytorch/resnet18_adam_256_bg_aug
