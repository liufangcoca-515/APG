#!/usr/bin/env python3

"""
测试脚本：加载预训练模型并在验证集上进行测试
"""

import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
import sys

import cfg
from func_3d import function
from func_3d.utils import get_network, create_logger
from func_3d.dataset import get_dataloader
from train_3d_apg import validation_sam_optimized, free_cuda_memory

def main():
    # 直接使用cfg.parse_args()获取配置
    args = cfg.parse_args()
    
    # 设置保存目录
    if not args.weights:
        raise ValueError("必须指定权重文件路径 (-weights)")
    
    model_dir = os.path.dirname(args.weights)
    save_dir = os.path.join(model_dir, 'test_results')
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置日志
    logger = create_logger(save_dir, phase='test')
    logger.info(f"Testing model: {args.weights}")
    logger.info(f"Results will be saved to: {save_dir}")
    
    # 设置设备
    device = torch.device(f'cuda:{args.gpu_device}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 重要：临时修改func_3d.utils中的device变量
    import func_3d.utils
    func_3d.utils.device = device
    
    # 加载模型
    net = get_network(args, args.net, use_gpu=True, gpu_device=device)
    
    # 使用bfloat16精度
    if hasattr(args, 'use_bfloat16') and args.use_bfloat16:
        net.to(dtype=torch.bfloat16)
    
    # 加载模型权重
    logger.info(f"Loading model weights from: {args.weights}")
    weights = torch.load(args.weights, map_location=device)
    
    # 检查权重是否是字典格式，并包含'model'键
    if isinstance(weights, dict) and 'model' in weights:
        weights = weights['model']
    
    # 加载权重
    net.load_state_dict(weights, strict=False)
    logger.info("Model weights loaded successfully")
    
    # 获取数据加载器
    _, test_loader = get_dataloader(args)
    
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
    args.path_helper = {'ckpt_path': save_dir}
    
    # 运行验证
    logger.info("Starting evaluation...")
    score, (iou, dice) = validation_sam_optimized(args, test_loader, 0, net)
    
    # 打印结果
    logger.info(f"Test results: IoU={iou:.4f}, Dice={dice:.4f}, Score={score:.4f}")
    
    # 保存结果到文件
    result_file = os.path.join(save_dir, 'test_results.txt')
    with open(result_file, 'w') as f:
        f.write(f"Model: {args.weights}\n")
        f.write(f"IoU: {iou:.4f}\n")
        f.write(f"Dice: {dice:.4f}\n")
        f.write(f"Score: {score:.4f}\n")
    
    logger.info(f"Results saved to {result_file}")
    
    # 检查CSV文件是否已生成
    csv_path = os.path.join(save_dir, 'validation_metrics.csv')
    class_csv_path = os.path.join(save_dir, 'validation_class_metrics.csv')
    
    if os.path.exists(csv_path) and os.path.exists(class_csv_path):
        logger.info(f"Detailed metrics saved to:")
        logger.info(f"  - {csv_path}")
        logger.info(f"  - {class_csv_path}")
    
    logger.info("Testing completed successfully!")

if __name__ == '__main__':
    main() 