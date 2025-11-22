#!/usr/bin/env python3

"""
测试脚本：加载预训练模型并在验证集上进行测试
"""

import os
import torch
import argparse
import numpy as np
from tqdm import tqdm

import cfg
from func_3d import function
from func_3d.utils import get_network, create_logger
from func_3d.dataset import get_dataloader
from train_3d_apg import validation_sam_optimized, free_cuda_memory

def parse_args():
    parser = argparse.ArgumentParser(description='测试预训练模型')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='预训练模型路径')
    parser.add_argument('--exp_name', type=str, default='med3d_btcv_wpretrain_continue_2025_07_14_11_12_10',
                        help='实验名称，用于查找配置')
    parser.add_argument('--gpu_device', type=int, default=0, 
                        help='GPU设备ID')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='结果保存目录，默认为模型目录下的test_results')
    parser.add_argument('--save_images', action='store_true', default=True,
                        help='是否保存可视化结果')
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 加载原始实验的配置
    original_args = cfg.parse_args(['--exp_name', args.exp_name])
    
    # 更新GPU设备ID
    original_args.gpu_device = args.gpu_device
    
    # 设置保存目录
    if args.save_dir is None:
        model_dir = os.path.dirname(args.model_path)
        args.save_dir = os.path.join(model_dir, 'test_results')
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(args.save_dir, 'test_log.txt')
    logger = create_logger(args.save_dir, log_file_name=os.path.basename(log_file))
    logger.info(f"Testing model: {args.model_path}")
    logger.info(f"Results will be saved to: {args.save_dir}")
    
    # 设置设备
    device = torch.device(f'cuda:{args.gpu_device}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 加载模型
    net = get_network(original_args, original_args.net, use_gpu=True, gpu_device=device)
    
    # 使用bfloat16精度
    if hasattr(original_args, 'use_bfloat16') and original_args.use_bfloat16:
        net.to(dtype=torch.bfloat16)
    
    # 加载模型权重
    logger.info(f"Loading model weights from: {args.model_path}")
    weights = torch.load(args.model_path, map_location=device)
    
    # 检查权重是否是字典格式，并包含'model'键
    if isinstance(weights, dict) and 'model' in weights:
        weights = weights['model']
    
    # 加载权重
    net.load_state_dict(weights, strict=False)
    logger.info("Model weights loaded successfully")
    
    # 获取数据加载器
    _, test_loader = get_dataloader(original_args)
    
    # 修改验证数据加载器的批量大小为1，减少内存使用
    from torch.utils.data import DataLoader
    test_dataset = test_loader.dataset
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=False
    )
    
    # 设置模型为评估模式
    net.eval()
    
    # 清理内存
    free_cuda_memory()
    
    # 创建保存结果的目录
    original_args.path_helper = {'ckpt_path': args.save_dir}
    
    # 运行验证
    logger.info("Starting evaluation...")
    score, (iou, dice) = validation_sam_optimized(original_args, test_loader, 0, net)
    
    # 打印结果
    logger.info(f"Test results: IoU={iou:.4f}, Dice={dice:.4f}, Score={score:.4f}")
    
    # 保存结果到文件
    result_file = os.path.join(args.save_dir, 'test_results.txt')
    with open(result_file, 'w') as f:
        f.write(f"Model: {args.model_path}\n")
        f.write(f"IoU: {iou:.4f}\n")
        f.write(f"Dice: {dice:.4f}\n")
        f.write(f"Score: {score:.4f}\n")
    
    logger.info(f"Results saved to {result_file}")
    
    # 检查CSV文件是否已生成
    csv_path = os.path.join(args.save_dir, 'validation_metrics.csv')
    class_csv_path = os.path.join(args.save_dir, 'validation_class_metrics.csv')
    
    if os.path.exists(csv_path) and os.path.exists(class_csv_path):
        logger.info(f"Detailed metrics saved to:")
        logger.info(f"  - {csv_path}")
        logger.info(f"  - {class_csv_path}")
    
    logger.info("Testing completed successfully!")

if __name__ == '__main__':
    main() 