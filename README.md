# CFDA-M
Codes and data for CFDA-M: Coarse-to-Fine Domain Adaptation for Mitochondria Segmentation via Patch-wise Image Alignment and Online Self-training (2022 IEEE International Conference on Bioinformatics and Biomedicine)
## Datasets
Data used in this paper can be download [here](https://pan.baidu.com/s/1loxVwzj0OeIw2OOQLrH36g?pwd=wf5f).
## Dependencies
## Training
### Coarse stage
```
cd /stage1
```
```
CUDA_VISIBLE_DEVICES=0 python train.py \
--name experiment_cut2seg \
--raw_A_dir ./preprocess/VNC2Lucchi/VNC/ \
--raw_A_seg_dir ./preprocess/VNC2Lucchi/VNC/ \
--raw_B_dir ./preprocess/VNC2Lucchi/Lucchi/ \
--sub_list_A ./preprocess/VNC2Lucchi/train_VNC.txt \
--sub_list_B ./preprocess/VNC2Lucchi/train_Lucchi.txt \
--batch_size 4 \
--angle 0 \
--model cut2seg_model_train \
--netG resnet_9blocks \
--netD basic \
--netS duseunet \
--pool_size 50 \
--no_dropout \
--dataset_mode cut2seg_train \
--input_nc 1  \
--output_nc 1 \
--output_nc_seg 1 \
--lambda_GAN 1.0 \
--lambda_NCE 1.0 \
--lambda_DICE 1.0 \
--lambda_SC 1.0 \
--checkpoints_dir ./checkpoints/VNC2Lucchi/ \
--display_id 0
```
### Fine stage
```
cd /stage2
```
```
CUDA_VISIBLE_DEVICES=0 python train.py
```
## Acknowledgement
This code is based on [AccSeg-Net](https://github.com/weih527/SSNS-Net) (MICCAI'21) by Bo Zhou et al. Should you have any further questions, please let us know. Thanks again for your interest.
