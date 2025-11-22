""" function for training and validation in one epoch
    Yunli Qi
"""

import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from monai.losses import DiceLoss, FocalLoss
from tqdm import tqdm
import numpy as np
import nibabel as nib

import cfg
from conf import settings
from func_3d.utils import eval_seg

args = cfg.parse_args()

class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=1, focal_weight=1):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss(to_onehot_y=True, sigmoid=True)
        self.focal_loss = FocalLoss(to_onehot_y=True, gamma=2.0)

    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        return self.dice_weight * dice + self.focal_weight * focal


GPUdevice = torch.device('cuda', args.gpu_device)
pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
paper_loss = CombinedLoss(dice_weight=1 / 21, focal_weight=20 / 21)
seed = torch.randint(1,11,(1,7))

torch.backends.cudnn.benchmark = True
scaler = torch.cuda.amp.GradScaler()
max_iterations = settings.EPOCH
dice_val_best = 0.0
global_step_best = 0
epoch_loss_values = []
metric_values = []



def train_sam(args, net: nn.Module, optimizer1, optimizer2, train_loader,
          epoch):
    hard = 0
    epoch_loss = 0
    epoch_prompt_loss = 0
    epoch_non_prompt_loss = 0
    ind = 0
    # train mode
    net.train()
    if optimizer1 is not None:
        optimizer1.zero_grad()
    if optimizer2 is not None:
        optimizer2.zero_grad()
    video_length = args.video_length

    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    prompt = args.prompt
    prompt_freq = args.prompt_freq

    lossfunc = criterion_G
    # lossfunc = paper_loss#.to(dtype=torch.bfloat16, device=GPUdevice)

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:
            torch.cuda.empty_cache()
            imgs_tensor = pack['image']
            mask_dict = pack['label']
            if prompt == 'click':
                pt_dict = pack['pt']
                point_labels_dict = pack['p_label']
            elif prompt == 'bbox':
                bbox_dict = pack['bbox']
            imgs_tensor = imgs_tensor.squeeze(0)
            imgs_tensor = imgs_tensor.to(dtype = torch.float32, device = GPUdevice)
            
            train_state = net.train_init_state(imgs_tensor=imgs_tensor)
            prompt_frame_id = list(range(0, video_length, prompt_freq))
            obj_list = []
            for id in prompt_frame_id:
                obj_list += list(mask_dict[id].keys())
            obj_list = list(set(obj_list))
            if len(obj_list) == 0:
                continue

            name = pack['image_meta_dict']['filename_or_obj']
            # reverse = np.random.rand() > 0.5

            with torch.cuda.amp.autocast():
                for id in prompt_frame_id:
                    for ann_obj_id in obj_list:
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
                                mask = torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice),
                            )
                video_segments = {}  # video_segments contains the per-frame segmentation results
            
                for out_frame_idx, out_obj_ids, out_mask_logits in net.train_propagate_in_video(train_state, start_frame_idx=0):
                    video_segments[out_frame_idx] = {
                        out_obj_id: out_mask_logits[i]
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                loss = 0
                non_prompt_loss = 0
                prompt_loss = 0
                for id in range(video_length):
                    for ann_obj_id in obj_list:
                        pred = video_segments[id][ann_obj_id]
                        pred = pred.unsqueeze(0)
                        # pred = torch.sigmoid(pred)
                        try:
                            mask = mask_dict[id][ann_obj_id].to(dtype = torch.float32, device = GPUdevice)
                        except KeyError:
                            mask = torch.zeros_like(pred).to(device=GPUdevice)
                        if args.train_vis:
                            os.makedirs(f'./temp/train/{name[0]}/{id}', exist_ok=True)
                            fig, ax = plt.subplots(1, 3)
                            ax[0].imshow(imgs_tensor[id, :, :, :].detach().cpu().permute(1, 2, 0).numpy().astype(int))
                            ax[0].axis('off')
                            ax[1].imshow(pred[0, 0, :, :].detach().cpu().numpy() > 0.5, cmap='gray')
                            ax[1].axis('off')
                            try:
                                bbox = bbox_dict[id][ann_obj_id]
                                ax[1].add_patch(plt.Rectangle((bbox[0][0], bbox[0][1]), bbox[0][2] - bbox[0][0], bbox[0][3] - bbox[0][1], edgecolor='green', facecolor=(0,0,0,0), lw=2))
                            except KeyError:
                                pass
                            ax[2].imshow(mask[0, 0, :, :].detach().cpu().numpy(), cmap='gray')
                            ax[2].axis('off')
                            plt.savefig(f'./temp/train/{name[0]}/{id}/{obj_list.index(ann_obj_id)}.png', bbox_inches='tight', pad_inches=0)
                            plt.close()
                        obj_loss = lossfunc(pred, mask)
                        loss += obj_loss.item()
                        if id in prompt_frame_id:
                            prompt_loss += obj_loss
                        else:
                            non_prompt_loss += obj_loss
                        
                        # 立即释放不需要的张量
                        del pred, mask, obj_loss
                        
                loss = loss / video_length / len(obj_list)
                if prompt_freq > 1:
                    non_prompt_loss = non_prompt_loss / (video_length - len(prompt_frame_id)) / len(obj_list)
                prompt_loss = prompt_loss / len(prompt_frame_id) / len(obj_list)

                pbar.set_postfix(**{'loss (batch)': loss})
                epoch_loss += loss
                epoch_prompt_loss += prompt_loss.item()
                if prompt_freq > 1:
                    epoch_non_prompt_loss += non_prompt_loss.item()

                # nn.utils.clip_grad_value_(net.parameters(), 0.1)
                if non_prompt_loss is not int and optimizer2 is not None and prompt_freq > 1:
                    non_prompt_loss.backward(retain_graph=True)
                    optimizer2.step()
                if optimizer1 is not None:
                    prompt_loss.backward()
                    optimizer1.step()
                
                    optimizer1.zero_grad()
                if optimizer2 is not None:
                    optimizer2.zero_grad()
                
            # --- 内存清理开始 ---
            # 重置网络内部状态
            net.reset_state(train_state)
            
            # 删除不再需要的Python变量引用，以便垃圾回收器回收显存
            del train_state, video_segments, imgs_tensor, mask_dict, loss, prompt_loss, non_prompt_loss
            if prompt == 'click':
                del pt_dict, point_labels_dict
            elif prompt == 'bbox':
                del bbox_dict
            
            # 清理PyTorch的CUDA缓存
            torch.cuda.empty_cache()
            # --- 内存清理结束 ---

            pbar.update()

    return epoch_loss / len(train_loader), epoch_prompt_loss / len(train_loader), epoch_non_prompt_loss / len(train_loader)

def validation_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):
     # eval mode
    net.eval()

    # 导入nibabel用于保存nii.gz格式
    import nibabel as nib
    
    n_val = len(val_loader)  # the number of batch
    mix_res = (0,)*1*2
    tot = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    prompt_freq = args.prompt_freq

    lossfunc = criterion_G
    # lossfunc = paper_loss

    prompt = args.prompt
    
    # 判断是否需要保存图片和nii.gz文件（每10个epoch保存一次）
    save_results = (epoch % 10 == 0) or (epoch == 0)
    
    # 创建保存当前epoch结果的文件夹
    save_dir = os.path.join(args.path_helper['ckpt_path'], f'val_results/epoch_{epoch}')
    if save_results and clean_dir and os.path.exists(save_dir):
        import shutil
        shutil.rmtree(save_dir)
    if save_results:
        os.makedirs(save_dir, exist_ok=True)
        # 添加日志信息
        print(f"Starting validation, results will be saved to: {save_dir}")
    else:
        print(f"Starting validation (without saving images and nii.gz files)")

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for pack in val_loader:
            imgs_tensor = pack['image']
            mask_dict = pack['label']
            if prompt == 'click':
                pt_dict = pack['pt']
                point_labels_dict = pack['p_label']
            elif prompt == 'bbox':
                bbox_dict = pack['bbox']
            if len(imgs_tensor.size()) == 5:
                imgs_tensor = imgs_tensor.squeeze(0)
            frame_id = list(range(imgs_tensor.size(0)))
            
            train_state = net.val_init_state(imgs_tensor=imgs_tensor)
            prompt_frame_id = list(range(0, len(frame_id), prompt_freq))
            obj_list = []
            for id in frame_id:
                obj_list += list(mask_dict[id].keys())
            obj_list = list(set(obj_list))
            if len(obj_list) == 0:
                continue

            name = pack['image_meta_dict']['filename_or_obj']
            # 打印当前处理的样本名称和帧数
            # print(f"处理样本: {name[0]}, 总帧数: {len(frame_id)}, 对象数: {len(obj_list)}")

            with torch.no_grad():
                for id in prompt_frame_id:
                    for ann_obj_id in obj_list:
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
                                mask = torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice),
                            )
                video_segments = {}  # video_segments contains the per-frame segmentation results
            
                for out_frame_idx, out_obj_ids, out_mask_logits in net.propagate_in_video(train_state, start_frame_idx=0):
                    video_segments[out_frame_idx] = {
                        out_obj_id: out_mask_logits[i]
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                
                # 打印网络推理结果信息
                # print(f"样本 {name[0]} 的video_segments包含的帧: {sorted(list(video_segments.keys()))}")
                # for frame_idx in sorted(list(video_segments.keys())):
                #     print(f"  帧 {frame_idx} 包含的对象ID: {sorted(list(video_segments[frame_idx].keys()))}")

                loss = 0
                pred_iou = 0
                pred_dice = 0
                saved_frames = set()  # 用于记录已保存的帧
                
                # 只在需要保存结果时执行以下代码
                if save_results:
                    # 确保为所有帧创建目录
                    sample_dir = f'{save_dir}/{name[0]}'
                    os.makedirs(sample_dir, exist_ok=True)
                    
                    # 为每一帧保存一张综合图片
                    for id in frame_id:  # 遍历所有帧
                        # 图像大小和对象数量决定图像布局
                        n_objects = len(obj_list)
                        if n_objects == 0:
                            continue
                            
                        # 创建一个大图，包含原图和所有对象的预测
                        if n_objects <= 3:
                            fig, axs = plt.subplots(n_objects, 3, figsize=(15, 5*n_objects))
                            # 处理只有一个对象的情况
                            if n_objects == 1:
                                axs = [axs]
                        else:
                            # 如果对象太多，使用网格布局
                            cols = 3
                            rows = (n_objects + cols - 1) // cols
                            fig, axs = plt.subplots(rows, cols*3, figsize=(15, 5*rows))
                            # 重塑axs以便于访问
                            axs = [axs[i//cols, (i%cols)*3:(i%cols)*3+3] for i in range(n_objects)]
                        
                        # 原始图像（作为参考）
                        orig_img = imgs_tensor[id].cpu().permute(1, 2, 0).numpy().astype(int)
                        
                        # 为每个对象生成预测和mask
                        for i, ann_obj_id in enumerate(obj_list):
                            try:
                                pred = video_segments[id][ann_obj_id]
                                pred = pred.unsqueeze(0)
                                
                                try:
                                    mask = mask_dict[id][ann_obj_id].to(dtype=torch.float32, device=GPUdevice)
                                except KeyError:
                                    mask = torch.zeros_like(pred).to(device=GPUdevice)
                                
                                # 显示原始图像
                                axs[i][0].imshow(orig_img)
                                axs[i][0].set_title(f'Frame {id} - Object {ann_obj_id}')
                                axs[i][0].axis('off')
                                
                                # 显示预测mask
                                pred_np = pred[0, 0].cpu().numpy() > 0.5
                                axs[i][1].imshow(pred_np, cmap='gray')
                                axs[i][1].set_title(f'Prediction Mask')
                                axs[i][1].axis('off')
                                
                                # 显示真实mask
                                gt_np = mask[0, 0].cpu().numpy()
                                axs[i][2].imshow(gt_np, cmap='gray')
                                axs[i][2].set_title(f'Ground Truth Mask')
                                axs[i][2].axis('off')
                                
                                # 计算评估指标
                                loss += lossfunc(pred, mask)
                                temp = eval_seg(pred, mask, threshold)
                                pred_iou += temp[0]
                                pred_dice += temp[1]
                            except KeyError as e:
                                print(f"警告: 处理帧 {id} 对象 {ann_obj_id} 时出现KeyError: {e}")
                                continue
                        
                        # 调整布局并保存
                        plt.tight_layout()
                        plt.savefig(f'{sample_dir}/frame_{id}.png')
                        plt.close(fig)
                        
                        saved_frames.add(id)
                    
                    # 检查是否有帧未被处理（可能是因为没有对象或处理出错）
                    missing_frames = set(frame_id) - saved_frames
                    if missing_frames:
                        print(f"警告: 样本 {name[0]} 中有 {len(missing_frames)} 帧未被保存: {missing_frames}")
                        
                        # 尝试为缺失的帧保存原始图像
                        for id in missing_frames:
                            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                            ax.imshow(imgs_tensor[id].cpu().permute(1, 2, 0).numpy().astype(int))
                            ax.set_title(f'Frame {id} (No Segmentation)')
                            ax.axis('off')
                            plt.savefig(f'{sample_dir}/frame_{id}_no_seg.png')
                            plt.close(fig)
                    
                    # 检查是否保存了所有帧
                    print(f"样本 {name[0]} 已保存 {len(saved_frames)}/{len(frame_id)} 帧")
                    
                    # 首先获取图像尺寸
                    img_shape = imgs_tensor[0].shape[1:]  # H, W
                    
                    # 为每一帧创建额外的对比可视化（红色=预测，绿色=真实标签）
                    for id in frame_id:
                        if id in saved_frames:  # 只处理成功保存的帧
                            orig_img = imgs_tensor[id].cpu().permute(1, 2, 0).numpy()
                            # 归一化到0-1范围用于显示
                            if orig_img.max() > 1.0:
                                orig_img = orig_img / 255.0
                            
                            # 创建单独的预测和真实标签叠加图
                            plt.figure(figsize=(15, 5))
                            
                            # 原图（参考）
                            plt.subplot(131)
                            plt.imshow(orig_img)
                            plt.title('Original Image')
                            plt.axis('off')
                            
                            # 预测结果叠加（红色）
                            plt.subplot(132)
                            plt.imshow(orig_img)
                            for ann_obj_id in obj_list:
                                try:
                                    pred = video_segments[id][ann_obj_id]
                                    pred_mask = (pred.cpu().numpy() > 0.5).reshape(img_shape)
                                    # 创建红色mask（半透明）
                                    red_mask = np.zeros((*img_shape, 4))
                                    red_mask[pred_mask, 0] = 1.0  # 红色通道
                                    red_mask[pred_mask, 3] = 0.5  # 透明度
                                    plt.imshow(red_mask)
                                except KeyError:
                                    pass
                            plt.title('Prediction Result (Red)')
                            plt.axis('off')
                            
                            # 真实标签叠加（绿色）
                            plt.subplot(133)
                            plt.imshow(orig_img)
                            for ann_obj_id in obj_list:
                                try:
                                    mask = mask_dict[id][ann_obj_id].to(dtype=torch.float32, device=GPUdevice)
                                    gt_mask = mask[0, 0].cpu().numpy() > 0.5
                                    # 创建绿色mask（半透明）
                                    green_mask = np.zeros((*img_shape, 4))
                                    green_mask[gt_mask, 1] = 1.0  # 绿色通道
                                    green_mask[gt_mask, 3] = 0.5  # 透明度
                                    plt.imshow(green_mask)
                                except KeyError:
                                    pass
                            plt.title('Ground Truth (Green)')
                            plt.axis('off')
                            
                            plt.tight_layout()
                            plt.savefig(f'{sample_dir}/frame_{id}_overlay.png')
                            plt.close()
                            
                            # 创建红绿对比图（红色=预测，绿色=真实标签，黄色=重叠区域）
                            plt.figure(figsize=(8, 8))
                            plt.imshow(orig_img)
                            
                            # 创建对比mask
                            comparison_mask = np.zeros((*img_shape, 4))
                            
                            for ann_obj_id in obj_list:
                                try:
                                    # 预测掩码（红色）
                                    pred = video_segments[id][ann_obj_id]
                                    pred_mask = (pred.cpu().numpy() > 0.5).reshape(img_shape)
                                    comparison_mask[pred_mask, 0] = 1.0  # 红色通道
                                    comparison_mask[pred_mask, 3] = 0.5  # 透明度
                                    
                                    # 真实标签（绿色）
                                    try:
                                        mask = mask_dict[id][ann_obj_id].to(dtype=torch.float32, device=GPUdevice)
                                        gt_mask = mask[0, 0].cpu().numpy() > 0.5
                                        comparison_mask[gt_mask, 1] = 1.0  # 绿色通道
                                        comparison_mask[gt_mask, 3] = 0.5  # 透明度
                                        # 注意：重叠区域会自动变为黄色（红+绿）
                                    except KeyError:
                                        pass
                                except KeyError:
                                    pass
                            
                            plt.imshow(comparison_mask)
                            plt.title('Prediction (Red) vs Ground Truth (Green)')
                            plt.axis('off')
                            plt.tight_layout()
                            plt.savefig(f'{sample_dir}/frame_{id}_comparison.png')
                            plt.close()
                    
                    # 创建3D体积用于保存为nii.gz格式
                    # img_shape已在前面定义
                    
                    # 为每个对象创建一个3D体积
                    for ann_obj_id in obj_list:
                        # 创建空的3D数组，大小为 [帧数, 高度, 宽度]
                        pred_volume = np.zeros((len(frame_id),) + img_shape, dtype=np.float32)
                        gt_volume = np.zeros((len(frame_id),) + img_shape, dtype=np.float32)
                        
                        # 填充预测和真实标签数据
                        for idx, id in enumerate(sorted(frame_id)):
                            try:
                                # 获取预测结果
                                pred = video_segments[id][ann_obj_id]
                                pred_np = pred.cpu().numpy() > 0.5
                                # 确保pred_np是2D的
                                pred_np = pred_np.reshape(img_shape)
                                pred_volume[idx] = pred_np
                                
                                # 获取真实标签
                                try:
                                    mask = mask_dict[id][ann_obj_id].to(dtype=torch.float32, device=GPUdevice)
                                    gt_np = mask[0, 0].cpu().numpy()
                                    gt_volume[idx] = gt_np
                                except KeyError:
                                    # 如果没有真实标签，保持为0
                                    pass
                            except KeyError:
                                # 如果特定帧没有对象，保持为0
                                print(f"警告: 帧 {id} 对象 {ann_obj_id} 没有预测结果")
                        
                        # 将预测结果保存为nii.gz格式
                        # 创建nibabel对象 - 使用标准方向矩阵，像素尺寸为1mm
                        affine = np.eye(4)
                        pred_nii = nib.Nifti1Image(pred_volume, affine)
                        gt_nii = nib.Nifti1Image(gt_volume, affine)
                        
                        # 保存文件
                        pred_nii_path = f'{sample_dir}/pred_obj_{ann_obj_id}.nii.gz'
                        gt_nii_path = f'{sample_dir}/gt_obj_{ann_obj_id}.nii.gz'
                        nib.save(pred_nii, pred_nii_path)
                        nib.save(gt_nii, gt_nii_path)
                        # print(f"已保存对象 {ann_obj_id} 的预测和真实标签为nii.gz格式")
                    
                    # 创建一个包含所有对象预测的综合体积
                    if obj_list:
                        # 创建一个标签体积，每个像素的值对应对象ID
                        combined_pred_volume = np.zeros((len(frame_id),) + img_shape, dtype=np.float32)
                        combined_gt_volume = np.zeros((len(frame_id),) + img_shape, dtype=np.float32)
                        
                        # 填充数据 - 使用对象ID作为像素值
                        for idx, id in enumerate(sorted(frame_id)):
                            pred_slice = combined_pred_volume[idx]  # 获取当前帧的切片
                            gt_slice = combined_gt_volume[idx]  # 获取当前帧的切片
                            
                            for ann_obj_id in obj_list:
                                try:
                                    # 获取预测结果
                                    pred = video_segments[id][ann_obj_id]
                                    pred_np = pred.cpu().numpy() > 0.5
                                    pred_np = pred_np.reshape(img_shape)  # 确保形状正确
                                    
                                    # 将预测区域的值设为对象ID
                                    pred_slice[pred_np] = ann_obj_id
                                    
                                    # 获取真实标签
                                    try:
                                        mask = mask_dict[id][ann_obj_id].to(dtype=torch.float32, device=GPUdevice)
                                        gt_np = mask[0, 0].cpu().numpy() > 0.5
                                        gt_slice[gt_np] = ann_obj_id
                                    except KeyError:
                                        pass
                                except KeyError:
                                    pass
                            
                            # 将修改后的切片放回体积
                            combined_pred_volume[idx] = pred_slice
                            combined_gt_volume[idx] = gt_slice
                        
                        # 保存综合体积
                        affine = np.eye(4)
                        combined_pred_nii = nib.Nifti1Image(combined_pred_volume, affine)
                        combined_gt_nii = nib.Nifti1Image(combined_gt_volume, affine)
                        
                        combined_pred_path = f'{sample_dir}/combined_pred.nii.gz'
                        combined_gt_path = f'{sample_dir}/combined_gt.nii.gz'
                        nib.save(combined_pred_nii, combined_pred_path)
                        nib.save(combined_gt_nii, combined_gt_path)
                        # print(f"已保存综合预测和真实标签为nii.gz格式")

                    # 创建一个用于存储原始图像的3D体积
                    # 由于图像是RGB格式，我们需要处理3个通道
                    orig_volume = np.zeros((len(frame_id), 3) + img_shape, dtype=np.float32)  # 修改为(N, C, H, W)格式
                    
                    # 填充原始图像数据
                    for idx, id in enumerate(sorted(frame_id)):
                        # 保持通道维度在第二维
                        orig_img = imgs_tensor[id].cpu().numpy()  # 这应该已经是0-255范围
                        orig_volume[idx] = orig_img
                    
                    # 将原始图像保存为nii.gz格式
                    orig_nii = nib.Nifti1Image(orig_volume, affine)
                    orig_nii_path = f'{sample_dir}/original.nii.gz'
                    nib.save(orig_nii, orig_nii_path)
                    # print(f"已保存原始图像为nii.gz格式，形状为: {orig_volume.shape}")
                else:
                    # 如果不保存图片，仍然需要计算评估指标
                    for id in frame_id:
                        for ann_obj_id in obj_list:
                            try:
                                pred = video_segments[id][ann_obj_id]
                                pred = pred.unsqueeze(0)
                                
                                try:
                                    mask = mask_dict[id][ann_obj_id].to(dtype=torch.float32, device=GPUdevice)
                                except KeyError:
                                    mask = torch.zeros_like(pred).to(device=GPUdevice)
                                
                                # 计算评估指标
                                loss += lossfunc(pred, mask)
                                temp = eval_seg(pred, mask, threshold)
                                pred_iou += temp[0]
                                pred_dice += temp[1]
                            except KeyError:
                                pass

                total_num = len(frame_id) * len(obj_list)
                loss = loss / total_num
                temp = (pred_iou / total_num, pred_dice / total_num)
                tot += loss

                mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

            net.reset_state(train_state)
            pbar.update()
    
    # 打印验证完成信息
    if save_results:
        print(f"验证完成，结果已保存到: {save_dir}")
    else:
        print(f"验证完成（未保存图片和nii.gz文件）")

    return tot/ n_val , tuple([a/n_val for a in mix_res])