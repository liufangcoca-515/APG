环境用


训练用conda activate SAMMED3D
处理数据用 conda activate medsam2

# 测试模型的指令
## -weights 模型路径
## -data_path 测试数据路径
LD_LIBRARY_PATH=/home/liufang882/anaconda3/envs/SAMMED3D/lib/python3.9/site-packages/torch/lib:/usr/local/cuda-12.5/targets/x86_64-linux/lib:$LD_LIBRARY_PATH /home/liufang882/anaconda3/envs/SAMMED3D/bin/python test_model.py -weights /home/liufang882/YOLO_SAM/Medical-SAM2-main/logs/apg_btcv_wpretrain_2025_10_24_17_31_20/Model/latest_epoch.pth -sam_config sam2_hiera_s -data_path ./btcv -gpu_device 0 -memory_bank_size 1

# 训练模型的指令

## exp_name是训练时保存的名称
## data_path是数据集路径，基本上直接./[路径]就可以
## -sam_ckpt 是预训练模型的位置
LD_LIBRARY_PATH=/home/liufang882/anaconda3/envs/SAMMED3D/lib/python3.9/site-packages/torch/lib:/usr/local/cuda-12.5/targets/x86_64-linux/lib:$LD_LIBRARY_PATH /home/liufang882/anaconda3/envs/SAMMED3D/bin/python train_3d_apg.py -net sam2 -exp_name apg_CCMASK_wpretrain -sam_config sam2_hiera_s -val_freq 10 -prompt bbox -prompt_freq 2 -gpu True -gpu_device 0 -data_path ./btcv -memory_bank_size 1 -sam_ckpt ./checkpoints/sam2_hiera_small.pt