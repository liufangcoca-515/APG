# train.py

""" train network using pytorch
    Yunli Qi
"""

import os
# 提前设置显存分配策略，避免碎片化导致的 OOM
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64,expandable_segments:True'
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
import time
import gc
import numpy as np

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

import cfg
from func_3d import function
from conf import settings
from func_3d.utils import get_network, set_log_dir, create_logger
from func_3d.dataset import get_dataloader
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import defaultdict

def free_cuda_memory():
    """主动释放CUDA内存，减少内存碎片"""
    # 清空PyTorch的CUDA缓存
    torch.cuda.empty_cache()
    
    # 强制Python的垃圾回收
    gc.collect()
    
    # 尝试释放更多的CUDA内存（适用于某些CUDA版本）
    try:
        torch.cuda.synchronize()
    except:
        pass
    
    # 打印当前内存使用情况
    if torch.cuda.is_available():
        print(f"CUDA Memory: "
              f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, "
              f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

def validation_sam_optimized(args, nice_test_loader, epoch, net, writer=None):
    """
    优化版的验证函数，添加内存管理策略
    """
    # 导入matplotlib并设置非交互式后端
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    import logging
    
    # 获取logger
    logger = logging.getLogger()
    
    # 内存管理策略设置
    # 1. 自适应下采样 - 根据图像大小和帧数自动决定是否下采样
    adaptive_downsample = True  # 启用自适应下采样
    
    # 2. 处理大型图像的策略
    max_frames_per_batch = 1    # 内存优化：每批仅处理1帧
    max_total_frames = 2000      # 如果总帧数超过此值，则启用采样
    sampling_strategy = 'uniform'  # 采样策略：'uniform'(均匀采样) 或 'random'(随机采样)
    
    # 3. 设置是否保存图像 - 可以完全禁用来节省内存
    save_images = True  # 启用图像保存
    
    logger.info(f"Starting validation for epoch {epoch}...")
    logger.info(f"Memory optimization settings: adaptive_downsample={adaptive_downsample}, max_frames_per_batch={max_frames_per_batch}")
    
    # 使用torch.no_grad()减少内存使用
    with torch.no_grad():
        # 确保在验证前释放不必要的内存
        free_cuda_memory()
        
        # 设置模型为评估模式
        net.eval()
        
        # 初始化指标
        total_iou = 0.0
        total_dice = 0.0
        case_num = 0
        
        # 为每个类别（对象ID）初始化指标统计
        class_dice_values = defaultdict(list)  # 每个类别的dice值列表
        class_iou_values = defaultdict(list)   # 每个类别的iou值列表
        class_counts = defaultdict(int)        # 每个类别出现的次数
        
        # 创建保存结果的目录
        save_path = os.path.join(args.path_helper['ckpt_path'], f'val_results/epoch_{epoch}')
        if save_images:
            os.makedirs(save_path, exist_ok=True)
        
        # 判断是否需要保存图片和nii.gz文件（每10个epoch保存一次）
        save_results = save_images and ((epoch % 10 == 0) or (epoch == 0))
        
        # 设置设备
        GPUdevice = torch.device('cuda:' + str(args.gpu_device))
        prompt = args.prompt
        prompt_freq = args.prompt_freq
        lossfunc = function.criterion_G
        threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
        
        logger.info(f"Processing {len(nice_test_loader)} validation samples...")
        
        # 使用小批量处理验证数据
        for i, data in enumerate(tqdm(nice_test_loader, desc="Validating", leave=False)):
            try:
                # 每处理一个样本后强制清理内存
                if i > 0:
                    free_cuda_memory()
                
                # 获取数据
                imgs_tensor = data['image']
                mask_dict = data['label']
                meta_info = data.get('image_meta_dict', None)
                
                # 处理张量维度
                if len(imgs_tensor.size()) == 5:
                    imgs_tensor = imgs_tensor.squeeze(0)
                frame_id = list(range(imgs_tensor.size(0)))
                
                # 获取样本名称
                name = meta_info['filename_or_obj']
                logger.info(f"Processing sample: {name[0]}, Total frames: {len(frame_id)}")
                
                # 自适应下采样 - 根据图像大小和帧数决定下采样比例
                enable_downsample = False
                downsample_factor = 1.0
                
                if adaptive_downsample:
                    # 获取图像尺寸
                    img_height, img_width = imgs_tensor.shape[-2], imgs_tensor.shape[-1]
                    total_frames = len(frame_id)
                    
                    # 根据图像尺寸和帧数计算内存需求
                    estimated_memory_mb = (img_height * img_width * total_frames * 4) / (1024 * 1024)  # 粗略估计，每像素4字节
                    logger.info(f"Estimated memory requirement: {estimated_memory_mb:.2f} MB")
                    
                    # 根据内存需求决定下采样比例
                    if estimated_memory_mb > 8000:  # 如果估计内存需求大于8GB
                        enable_downsample = True
                        downsample_factor = 0.25  # 激进下采样
                        logger.info(f"Using aggressive downsampling (factor: {downsample_factor})")
                    elif estimated_memory_mb > 4000:  # 如果估计内存需求大于4GB
                        enable_downsample = True
                        downsample_factor = 0.5   # 中等下采样
                        logger.info(f"Using medium downsampling (factor: {downsample_factor})")
                    elif estimated_memory_mb > 2000:  # 如果估计内存需求大于2GB
                        enable_downsample = True
                        downsample_factor = 0.75  # 轻微下采样
                        logger.info(f"Using light downsampling (factor: {downsample_factor})")
                
                # 如果启用下采样，对图像进行下采样处理
                if enable_downsample:
                    # 记录原始尺寸
                    original_size = imgs_tensor.shape
                    # 使用插值下采样图像
                    imgs_tensor = torch.nn.functional.interpolate(
                        imgs_tensor.reshape(-1, 3, original_size[-2], original_size[-1]),
                        scale_factor=downsample_factor,
                        mode='bilinear',
                        align_corners=False
                    ).reshape(original_size[0], original_size[1], 3, 
                              int(original_size[-2]*downsample_factor), 
                              int(original_size[-1]*downsample_factor))
                    
                    # 同样下采样mask
                    for frame_id_key in mask_dict.keys():
                        for obj_id in mask_dict[frame_id_key].keys():
                            mask = mask_dict[frame_id_key][obj_id]
                            mask_dict[frame_id_key][obj_id] = torch.nn.functional.interpolate(
                                mask, 
                                scale_factor=downsample_factor,
                                mode='nearest'
                            )
                
                # 处理点击或边界框提示
                if prompt == 'click':
                    pt_dict = data['pt']
                    point_labels_dict = data['p_label']
                    
                    # 如果启用了下采样，需要调整点的坐标
                    if enable_downsample:
                        for frame_id_key in pt_dict.keys():
                            for obj_id in pt_dict[frame_id_key].keys():
                                pt_dict[frame_id_key][obj_id] = pt_dict[frame_id_key][obj_id] * downsample_factor
                                
                elif prompt == 'bbox':
                    bbox_dict = data['bbox']
                    
                    # 如果启用了下采样，需要调整边界框的坐标
                    if enable_downsample:
                        for frame_id_key in bbox_dict.keys():
                            for obj_id in bbox_dict[frame_id_key].keys():
                                bbox_dict[frame_id_key][obj_id] = bbox_dict[frame_id_key][obj_id] * downsample_factor
                
                # 处理大型图像 - 如果帧数过多，采用采样策略
                selected_frames = frame_id
                if len(frame_id) > max_total_frames:
                    logger.info(f"Large image detected ({len(frame_id)} frames). Using frame sampling.")
                    if sampling_strategy == 'uniform':
                        # 均匀采样
                        step = max(1, len(frame_id) // max_total_frames)
                        selected_frames = frame_id[::step][:max_total_frames]
                    else:  # 'random'
                        # 随机采样
                        import random
                        random.seed(42)  # 固定随机种子以保证可重复性
                        selected_frames = sorted(random.sample(frame_id, min(max_total_frames, len(frame_id))))
                    
                    logger.info(f"Selected {len(selected_frames)} frames for processing")
                
                # 初始化网络状态
                train_state = net.val_init_state(imgs_tensor=imgs_tensor)
                prompt_frame_id = list(range(0, len(selected_frames), prompt_freq))
                if not prompt_frame_id:
                    prompt_frame_id = [0]  # 确保至少有一个提示帧
                
                # 获取所有对象ID
                obj_list = []
                for id in selected_frames:
                    if id in mask_dict:
                        obj_list += list(mask_dict[id].keys())
                obj_list = list(set(obj_list))
                
                # 如果没有对象，跳过当前样本
                if len(obj_list) == 0:
                    logger.info(f"No objects found in sample {name[0]}, skipping.")
                    continue
                
                logger.info(f"Found {len(obj_list)} objects in sample {name[0]}: {sorted(obj_list)}")
                
                # 添加提示点
                for id in prompt_frame_id:
                    # 分批处理对象以减少内存使用
                    obj_batch_size = 1  # 每次处理一个对象
                    for obj_start_idx in range(0, len(obj_list), obj_batch_size):
                        obj_end_idx = min(obj_start_idx + obj_batch_size, len(obj_list))
                        batch_objs = obj_list[obj_start_idx:obj_end_idx]
                        
                        for ann_obj_id in batch_objs:
                            try:
                                if prompt == 'click':
                                    points = pt_dict[id][ann_obj_id].to(device=GPUdevice)
                                    labels = point_labels_dict[id][ann_obj_id].to(device=GPUdevice)
                                    _, _, _ = net.train_add_new_points(
                                        inference_state=train_state,
                                        frame_idx=id,
                                        obj_id=ann_obj_id,
                                        points=points,
                                        labels=labels,
                                        clear_old_points=False,
                                    )
                                elif prompt == 'bbox':
                                    bbox = bbox_dict[id][ann_obj_id]
                                    _, _, _ = net.train_add_new_bbox(
                                        inference_state=train_state,
                                        frame_idx=id,
                                        obj_id=ann_obj_id,
                                        bbox=bbox.to(device=GPUdevice),
                                        clear_old_points=False,
                                    )
                            except KeyError:
                                _, _, _ = net.train_add_new_mask(
                                    inference_state=train_state,
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    mask=torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice),
                                )
                            
                            # 每添加一个对象后清理内存
                            torch.cuda.empty_cache()
                
                # 分批处理视频帧以节省内存
                video_segments = {}
                
                # 分批获取网络推理结果 - 使用更小的批量
                for start_idx in range(0, len(selected_frames), max_frames_per_batch):
                    end_idx = min(start_idx + max_frames_per_batch, len(selected_frames))
                    batch_frames = selected_frames[start_idx:end_idx]
                    
                    logger.info(f"Processing frames {batch_frames[0]}-{batch_frames[-1]} ({len(batch_frames)} frames)")
                    
                    try:
                        # 为当前批次的帧获取分割结果
                        for out_frame_idx, out_obj_ids, out_mask_logits in net.propagate_in_video(
                            train_state, 
                            start_frame_idx=batch_frames[0]
                        ):
                            # 只处理当前批次的帧
                            if out_frame_idx in batch_frames:
                                video_segments[out_frame_idx] = {
                                    out_obj_id: out_mask_logits[i]
                                    for i, out_obj_id in enumerate(out_obj_ids)
                                }
                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e):
                            logger.warning(f"CUDA OOM during frame processing. Skipping batch {start_idx}-{end_idx}.")
                            # 尝试释放内存并继续
                            free_cuda_memory()
                            continue
                        else:
                            raise e
                    
                    # 每批处理后清理缓存
                    free_cuda_memory()
                
                # 计算评估指标
                loss = 0
                pred_iou = 0
                pred_dice = 0
                valid_frames = 0
                
                # 为当前样本初始化每个类别的指标
                sample_class_dice = defaultdict(list)
                sample_class_iou = defaultdict(list)
                
                # 分批计算指标
                for start_idx in range(0, len(selected_frames), max_frames_per_batch):
                    end_idx = min(start_idx + max_frames_per_batch, len(selected_frames))
                    batch_frames = selected_frames[start_idx:end_idx]
                    
                    for id in batch_frames:
                        # 检查该帧是否在video_segments中
                        if id not in video_segments:
                            continue
                            
                        for ann_obj_id in obj_list:
                            try:
                                # 检查该对象是否在当前帧的结果中
                                if ann_obj_id not in video_segments[id]:
                                    continue
                                    
                                pred = video_segments[id][ann_obj_id]
                                pred = pred.unsqueeze(0)
                                
                                try:
                                    mask = mask_dict[id][ann_obj_id].to(dtype=torch.float32, device=GPUdevice)
                                except KeyError:
                                    mask = torch.zeros_like(pred).to(device=GPUdevice)
                                
                                # 计算评估指标
                                loss += lossfunc(pred, mask).item()
                                temp = function.eval_seg(pred, mask, threshold)
                                
                                # 获取IoU和Dice值
                                iou_val = temp[0]
                                dice_val = temp[1]
                                
                                # 记录每个类别的指标
                                sample_class_dice[ann_obj_id].append(dice_val)
                                sample_class_iou[ann_obj_id].append(iou_val)
                                
                                pred_iou += iou_val
                                pred_dice += dice_val
                                valid_frames += 1
                                
                                # 保存结果（如果需要）
                                if save_results and id % 10 == 0:  # 只保存每10帧，减少存储需求
                                    # 创建样本目录
                                    sample_dir = os.path.join(save_path, name[0])
                                    os.makedirs(sample_dir, exist_ok=True)
                                    
                                    try:
                                        # 1. 保存原始图像、预测mask和真实mask的并排比较
                                        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                                        
                                        # 显示原始图像
                                        orig_img = imgs_tensor[id].cpu().permute(1, 2, 0).numpy()
                                        # 归一化图像用于显示
                                        if orig_img.max() > 1.0:
                                            orig_img = orig_img / 255.0
                                        axs[0].imshow(orig_img)
                                        axs[0].set_title(f'Frame {id} - Object {ann_obj_id}')
                                        axs[0].axis('off')
                                        
                                        # 显示预测mask
                                        # 先将logits转换为概率再阈值化
                                        pred_np = (torch.sigmoid(pred)[0, 0].cpu().numpy() > 0.5)
                                        axs[1].imshow(pred_np, cmap='gray', vmin=0, vmax=1)
                                        axs[1].set_title(f'Prediction Mask (Dice={temp[1]:.4f})')
                                        axs[1].axis('off')
                                        
                                        # 显示真实mask
                                        gt_np = mask[0, 0].cpu().numpy() > 0.5
                                        axs[2].imshow(gt_np, cmap='gray', vmin=0, vmax=1)
                                        axs[2].set_title(f'Ground Truth Mask')
                                        axs[2].axis('off')
                                        
                                        # 保存并排比较图
                                        plt.tight_layout()
                                        plt.savefig(f'{sample_dir}/frame_{id}_obj_{ann_obj_id}_comparison.png', dpi=100)
                                        plt.close(fig)
                                        
                                        # 2. 保存叠加图（overlay）
                                        fig, ax = plt.subplots(figsize=(8, 8))
                                        ax.imshow(orig_img)
                                        
                                        # 创建掩码叠加层：红色=预测，绿色=真实，黄色=重叠
                                        overlay = np.zeros((*pred_np.shape, 4))
                                        # 预测区域为红色
                                        overlay[pred_np, 0] = 1.0  # R
                                        overlay[pred_np, 3] = 0.5  # Alpha
                                        # Ground truth area in green
                                        overlay[gt_np, 1] = 1.0    # G
                                        overlay[gt_np, 3] = 0.5    # Alpha
                                        
                                        ax.imshow(overlay)
                                        ax.set_title(f'Overlay - Red:Prediction, Green:Ground Truth, Yellow:Overlap')
                                        ax.axis('off')
                                        plt.tight_layout()
                                        plt.savefig(f'{sample_dir}/frame_{id}_obj_{ann_obj_id}_overlay.png', dpi=100)
                                        plt.close(fig)
                                        
                                    except Exception as viz_error:
                                        logger.error(f"Error saving visualization: {viz_error}")
                                
                                # 立即释放不需要的张量
                                del pred, mask
                                torch.cuda.empty_cache()
                            except KeyError:
                                pass
                            except Exception as e:
                                logger.error(f"Error processing object {ann_obj_id} in frame {id}: {e}")
                                continue
                
                # 计算当前样本每个类别的平均指标
                sample_class_avg_dice = {}
                sample_class_avg_iou = {}
                
                for obj_id in sample_class_dice:
                    if sample_class_dice[obj_id]:
                        avg_dice = sum(sample_class_dice[obj_id]) / len(sample_class_dice[obj_id])
                        sample_class_avg_dice[obj_id] = avg_dice
                        
                        # 更新全局类别统计
                        class_dice_values[obj_id].append(avg_dice)
                        class_counts[obj_id] += 1
                
                for obj_id in sample_class_iou:
                    if sample_class_iou[obj_id]:
                        avg_iou = sum(sample_class_iou[obj_id]) / len(sample_class_iou[obj_id])
                        sample_class_avg_iou[obj_id] = avg_iou
                        
                        # 更新全局类别统计
                        class_iou_values[obj_id].append(avg_iou)
                
                # 打印当前样本每个类别的Dice值
                logger.info(f"Sample {name[0]} class-wise Dice values:")
                for obj_id in sorted(sample_class_avg_dice.keys()):
                    logger.info(f"  Class {obj_id}: Dice={sample_class_avg_dice[obj_id]:.4f}, IoU={sample_class_avg_iou.get(obj_id, 0):.4f}")
                
                # 计算平均指标
                if valid_frames > 0:
                    avg_loss = loss / valid_frames
                    avg_iou = pred_iou / valid_frames
                    avg_dice = pred_dice / valid_frames
                    
                    total_iou += avg_iou
                    total_dice += avg_dice
                    case_num += 1
                    
                    logger.info(f"Sample {name[0]}: IoU={avg_iou:.4f}, Dice={avg_dice:.4f}, Frames={valid_frames}")
                
                # 重置网络状态并清理内存
                net.reset_state(train_state)
                del train_state, video_segments, imgs_tensor, mask_dict
                if prompt == 'click':
                    del pt_dict, point_labels_dict
                elif prompt == 'bbox':
                    del bbox_dict
                free_cuda_memory()
                
            except Exception as e:
                logger.error(f"Error processing batch {i}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                free_cuda_memory()  # 出错时也清理内存
                continue
        
        # 计算总体平均指标
        if case_num > 0:
            final_iou = total_iou / case_num
            final_dice = total_dice / case_num
            final_score = final_iou + final_dice
        else:
            final_iou = 0
            final_dice = 0
            final_score = 0
        
        # 计算每个类别的平均指标
        class_avg_dice = {}
        class_avg_iou = {}
        
        for obj_id in class_dice_values:
            if class_dice_values[obj_id]:
                class_avg_dice[obj_id] = sum(class_dice_values[obj_id]) / len(class_dice_values[obj_id])
        
        for obj_id in class_iou_values:
            if class_iou_values[obj_id]:
                class_avg_iou[obj_id] = sum(class_iou_values[obj_id]) / len(class_iou_values[obj_id])
        
        # 打印每个类别的平均指标
        logger.info(f"\nClass-wise metrics across all samples:")
        for obj_id in sorted(class_avg_dice.keys()):
            logger.info(f"  Class {obj_id}: Dice={class_avg_dice[obj_id]:.4f}, IoU={class_avg_iou.get(obj_id, 0):.4f}, Count={class_counts[obj_id]}")
        
        # 打印总体验证结果
        logger.info(f"\nValidation Epoch {epoch}: Avg IoU={final_iou:.4f}, Avg Dice={final_dice:.4f}, Cases={case_num}")
        
        # 记录到TensorBoard
        if writer:
            writer.add_scalar('Val/IoU', final_iou, epoch)
            writer.add_scalar('Val/Dice', final_dice, epoch)
            
            # 记录每个类别的指标
            for obj_id in class_avg_dice:
                writer.add_scalar(f'Val/Class_{obj_id}_Dice', class_avg_dice[obj_id], epoch)
            
            for obj_id in class_avg_iou:
                writer.add_scalar(f'Val/Class_{obj_id}_IoU', class_avg_iou[obj_id], epoch)
        
        # 将验证指标保存到CSV文件中
        import csv
        
        # 创建CSV文件路径
        csv_path = os.path.join(args.path_helper['ckpt_path'], 'validation_metrics.csv')
        csv_exists = os.path.exists(csv_path)
        
        # 写入总体指标到CSV文件
        with open(csv_path, 'a', newline='') as f:
            writer_csv = csv.writer(f)
            if not csv_exists:
                # 写入表头
                writer_csv.writerow(['Epoch', 'IoU', 'Dice', 'Score', 'Cases'])
            # 写入当前epoch的指标
            writer_csv.writerow([epoch, final_iou, final_dice, final_score, case_num])
        
        # 创建类别指标CSV文件路径
        class_csv_path = os.path.join(args.path_helper['ckpt_path'], 'validation_class_metrics.csv')
        class_csv_exists = os.path.exists(class_csv_path)
        
        # 写入每个类别的指标到CSV文件
        with open(class_csv_path, 'a', newline='') as f:
            writer_csv = csv.writer(f)
            if not class_csv_exists:
                # 写入表头
                header = ['Epoch']
                for obj_id in sorted(class_avg_dice.keys()):
                    header.extend([f'Class_{obj_id}_Dice', f'Class_{obj_id}_IoU', f'Class_{obj_id}_Count'])
                writer_csv.writerow(header)
            
            # 写入当前epoch的类别指标
            row = [epoch]
            for obj_id in sorted(class_avg_dice.keys()):
                row.extend([
                    class_avg_dice.get(obj_id, 0),
                    class_avg_iou.get(obj_id, 0),
                    class_counts.get(obj_id, 0)
                ])
            writer_csv.writerow(row)
        
        logger.info(f"Validation metrics saved to {csv_path} and {class_csv_path}")
        
        return final_score, (final_iou, final_dice)

def main():
    # 设置更激进的内存回收
    import gc
    gc.set_threshold(100, 5, 5)
    
    args = cfg.parse_args()
    
    # 减小内存库大小，避免显存碎片
    if not hasattr(args, 'memory_bank_size'):
        args.memory_bank_size = 1  # 设置默认值为1
    else:
        # 若用户传入值过大，自动降到1
        if args.memory_bank_size > 1:
            print(f"memory_bank_size 由 {args.memory_bank_size} 自动降为 1 以防止OOM")
            args.memory_bank_size = 1
    
    # 启用梯度检查点功能
    if not hasattr(args, 'use_gradient_checkpointing'):
        args.use_gradient_checkpointing = True  # 默认启用梯度检查点
    
    # 设置bfloat16精度
    if not hasattr(args, 'use_bfloat16'):
        args.use_bfloat16 = True  # 默认使用bfloat16精度

    GPUdevice = torch.device('cuda', args.gpu_device)

    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
    
    # 使用bfloat16精度
    if args.use_bfloat16:
        net.to(dtype=torch.bfloat16)
        
    if args.pretrain:
        print(f"Loading pretrained model from: {args.pretrain}")
        weights = torch.load(args.pretrain)
        # 检查权重是否是字典格式，并包含'model'键
        if isinstance(weights, dict) and 'model' in weights:
            weights = weights['model']
        net.load_state_dict(weights, strict=False)
        print("Pretrained model loaded successfully")

    # 设置日志目录和创建logger
    args.path_helper = set_log_dir('logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)
    
    # 确保日志文件被正确创建
    log_file_path = os.path.join(args.path_helper['log_path'], f"{time.strftime('%Y-%m-%d-%H-%M')}_train.log")
    logger.info(f"Log file created at: {log_file_path}")

    # 配置优化器
    sam_layers = (
                  []
                #   + list(net.image_encoder.parameters())
                #   + list(net.sam_prompt_encoder.parameters())
                  + list(net.sam_mask_decoder.parameters())
                  )
    mem_layers = (
                  []
                  + list(net.obj_ptr_proj.parameters())
                  + list(net.memory_encoder.parameters())
                  + list(net.memory_attention.parameters())
                  + list(net.mask_downsample.parameters())
                  )
    if len(sam_layers) == 0:
        optimizer1 = None
    else:
        optimizer1 = optim.Adam(sam_layers, lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    if len(mem_layers) == 0:
        optimizer2 = None
    else:
        optimizer2 = optim.Adam(mem_layers, lr=1e-8, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    
    # 配置CUDA优化
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # 获取数据加载器
    nice_train_loader, nice_test_loader = get_dataloader(args)
    
    # 强制将训练和验证的批次大小都设为1，以解决OOM问题
    from torch.utils.data import DataLoader
    train_dataset = nice_train_loader.dataset
    nice_train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,  # 训练时需要打乱数据
        num_workers=2, # 使用2个worker加载数据
        pin_memory=False # 关闭内存锁定
    )

    test_dataset = nice_test_loader.dataset
    nice_test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=False
    )

    # 设置检查点路径和tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.makedirs(settings.LOG_DIR, exist_ok=True)
    
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))
    
    checkpoint_path = os.path.join(args.path_helper['ckpt_path'], '{net}-{epoch}-{type}.pth')
    logger.info(f"Checkpoint path: {checkpoint_path}")

    # 开始训练
    best_score = 0.0
    best_epoch = 0
    
    logger.info("Starting training...")
    
    for epoch in range(settings.EPOCH):
        # 训练前清理内存
        free_cuda_memory()
        
        # 训练阶段
        net.train()
        time_start = time.time()
        loss, prompt_loss, non_prompt_loss = function.train_sam(args, net, optimizer1, optimizer2, nice_train_loader, epoch)
        time_end = time.time()
        
        logger.info(f'Train loss: {loss}, {prompt_loss}, {non_prompt_loss} || @ epoch {epoch}.')
        logger.info(f'Training time: {time_end - time_start:.2f} seconds')
        
        # 清理内存
        free_cuda_memory()
        
        # 验证阶段
        net.eval()
        # if epoch % args.val_freq == 0 or epoch == settings.EPOCH-1:
        if None:
            # 使用优化后的验证函数
            tol, (eiou, edice) = validation_sam_optimized(args, nice_test_loader, epoch, net, writer)
            
            logger.info(f'Validation results - Total score: {tol}, IoU: {eiou}, Dice: {edice} || @ epoch {epoch}.')
            
            # 记录到TensorBoard
            writer.add_scalar('Train/Loss', loss, epoch)
            writer.add_scalar('Train/PromptLoss', prompt_loss, epoch)
            writer.add_scalar('Train/NonPromptLoss', non_prompt_loss, epoch)

            # 保存模型
            is_best = tol > best_score
            if is_best:
                best_score = tol
                best_epoch = epoch
                logger.info(f'New best model found at epoch {epoch} with score {best_score:.4f}')
                torch.save({'model': net.state_dict()}, os.path.join(args.path_helper['ckpt_path'], 'best_model.pth'))
            
            # 保存最新模型
            torch.save({'model': net.state_dict()}, os.path.join(args.path_helper['ckpt_path'], 'latest_epoch.pth'))
            
            # 清理内存
            free_cuda_memory()
        else:
            # 即使不验证，也记录训练损失到TensorBoard
            writer.add_scalar('Train/Loss', loss, epoch)
            writer.add_scalar('Train/PromptLoss', prompt_loss, epoch)
            writer.add_scalar('Train/NonPromptLoss', non_prompt_loss, epoch)
            
            # 每10个epoch保存一次模型，即使不验证
            if epoch % 10 == 9:
                torch.save({'model': net.state_dict()}, os.path.join(args.path_helper['ckpt_path'], 'latest_epoch.pth'))
                logger.info(f'Model saved at epoch {epoch}')
    
    logger.info(f'Training completed. Best score: {best_score:.4f} at epoch {best_epoch}')
    writer.close()


if __name__ == '__main__':
    main()