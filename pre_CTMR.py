# -*- coding: utf-8 -*-
# %% import packages
# pip install connected-components-3d
import numpy as np

# import nibabel as nib
import SimpleITK as sitk
import os
import sys
import logging
import time
from datetime import datetime

join = os.path.join
from skimage import transform
from tqdm import tqdm
import cc3d

# 设置调试日志
debug_dir = "debug_logs"
os.makedirs(debug_dir, exist_ok=True)
log_file = join(debug_dir, f"pre_CTMR_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# 配置日志记录器
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("="*50)
logger.info("开始运行预处理脚本")
logger.info("="*50)

# convert nii image to npz files, including original image and corresponding masks
import argparse

# Command line arguments to make the script reusable for different dataset folders
parser = argparse.ArgumentParser(description='Preprocess NIfTI images to npz and slice npy files')
parser.add_argument('--nii_path', default='/home/liufang882/YOLO_SAM/APG/CC-Mask/imageTr', help='path to the nii images')
parser.add_argument('--gt_path', default='/home/liufang882/YOLO_SAM/APG/CC-Mask/label', help='path to the ground truth labels')
parser.add_argument('--npy_path', default='', help='output base path for npy files (if empty will use data/npy/CCMASK_test/<modality>_<anatomy>)')
parser.add_argument('--modality', default='', help='modality prefix used in npy output folder name')
parser.add_argument('--anatomy', default='', help='anatomy/dataset name used in npy output folder name')
parser.add_argument('--img_name_suffix', default='.nii.gz', help='image file suffix')
parser.add_argument('--gt_name_suffix', default='.nii.gz', help='ground-truth file suffix')
args = parser.parse_args()

modality = args.modality
anatomy = args.anatomy  # anantomy + dataset name
img_name_suffix = args.img_name_suffix
gt_name_suffix = args.gt_name_suffix
prefix = modality + "_" + anatomy + "_"

nii_path = args.nii_path
gt_path = args.gt_path
if args.npy_path:
    npy_path = args.npy_path
else:
    # default npy path matches prior behavior when no args provided
    npy_path = "data/npy/CCMASK_test/" + prefix[:-1]

# 记录路径信息
logger.info(f"图像路径: {nii_path}")
logger.info(f"标签路径: {gt_path}")
logger.info(f"输出路径: {npy_path}")

# 检查路径是否存在
if not os.path.exists(nii_path):
    logger.error(f"图像路径不存在: {nii_path}")
    sys.exit(1)
if not os.path.exists(gt_path):
    logger.error(f"标签路径不存在: {gt_path}")
    sys.exit(1)

try:
    os.makedirs(join(npy_path, "gts"), exist_ok=True)
    os.makedirs(join(npy_path, "imgs"), exist_ok=True)
    logger.info("成功创建输出目录")
except Exception as e:
    logger.error(f"创建输出目录失败: {str(e)}")
    sys.exit(1)

image_size = 1024
voxel_num_thre2d = 100
voxel_num_thre3d = 100

# 记录处理参数
logger.info(f"图像处理尺寸: {image_size}x{image_size}")
logger.info(f"3D小目标过滤阈值: {voxel_num_thre3d}")
logger.info(f"2D小目标过滤阈值: {voxel_num_thre2d}")

# 获取标签文件列表
try:
    names = sorted(os.listdir(gt_path))
    logger.info(f"标签目录中的文件数量: {len(names)}")
    
    if len(names) == 0:
        logger.error("标签目录为空")
        sys.exit(1)
        
    # 输出前几个文件名，用于调试
    logger.info(f"示例标签文件: {names[:3] if len(names) >= 3 else names}")
except Exception as e:
    logger.error(f"读取标签目录时出错: {str(e)}")
    sys.exit(1)

# 检查配对文件是否存在
valid_names = []
for name in names:
    try:
        image_name = name.split(gt_name_suffix)[0] + img_name_suffix
        img_path = join(nii_path, image_name)
        
        if not os.path.exists(img_path):
            logger.warning(f"找不到匹配的图像文件: {img_path}")
            continue
            
        valid_names.append(name)
    except Exception as e:
        logger.warning(f"处理文件 {name} 时出错: {str(e)}")

logger.info(f"有效文件对数量: {len(valid_names)}")
names = valid_names

if len(names) == 0:
    logger.error("没有找到有效的文件对")
    sys.exit(1)

# set label ids that are excluded
remove_label_ids = [
    12
]  # remove deodenum since it is scattered in the image, which is hard to specify with the bounding box
tumor_id = None  # only set this when there are multiple tumors; convert semantic masks to instance masks
# set window level and width
# https://radiopaedia.org/articles/windowing-ct
WINDOW_LEVEL = 40  # only for CT images
WINDOW_WIDTH = 400  # only for CT images

logger.info(f"要移除的标签ID: {remove_label_ids}")
logger.info(f"肿瘤ID设置: {tumor_id}")
logger.info(f"窗位设置 - 级别: {WINDOW_LEVEL}, 宽度: {WINDOW_WIDTH}")

# 统计信息
total_cases = len(names)
processed_cases = 0
empty_cases = 0
failed_cases = 0

logger.info(f"准备处理的案例数量: {total_cases}")
logger.info("="*50)

# %% save preprocessed images and masks as npz files
start_time = time.time()

for idx, name in enumerate(tqdm(names[:total_cases], desc="处理案例")):
    case_start_time = time.time()
    logger.info(f"[{idx+1}/{total_cases}] 开始处理案例: {name}")
    
    try:
        # 准备文件名
        image_name = name.split(gt_name_suffix)[0] + img_name_suffix
        gt_name = name
        base_name = gt_name.split(gt_name_suffix)[0]
        
        # 读取标签文件
        try:
            logger.info(f"读取标签文件: {join(gt_path, gt_name)}")
            gt_sitk = sitk.ReadImage(join(gt_path, gt_name))
            gt_data_ori = np.uint8(sitk.GetArrayFromImage(gt_sitk))
            
            # 记录原始标签信息
            unique_labels = np.unique(gt_data_ori)
            non_zero_count = np.sum(gt_data_ori > 0)
            logger.info(f"标签数据形状: {gt_data_ori.shape}")
            logger.info(f"标签中的唯一值: {unique_labels}")
            logger.info(f"非零像素总数: {non_zero_count}")
            
            if non_zero_count == 0:
                logger.warning("标签文件中没有非零像素")
        except Exception as e:
            logger.error(f"读取标签文件失败: {str(e)}")
            failed_cases += 1
            continue
        
        # 移除指定标签ID
        for remove_label_id in remove_label_ids:
            pixels_to_remove = np.sum(gt_data_ori == remove_label_id)
            gt_data_ori[gt_data_ori == remove_label_id] = 0
            logger.info(f"移除标签ID {remove_label_id}: {pixels_to_remove} 个像素")
        
        # 记录移除标签后的状态
        non_zero_after_removal = np.sum(gt_data_ori > 0)
        logger.info(f"移除标签后非零像素数: {non_zero_after_removal}")
        
        # 处理肿瘤标签
        if tumor_id is not None:
            try:
                tumor_bw = np.uint8(gt_data_ori == tumor_id)
                tumor_pixels = np.sum(tumor_bw)
                logger.info(f"肿瘤标签(ID={tumor_id})像素数: {tumor_pixels}")
                
                gt_data_ori[tumor_bw > 0] = 0
                # label tumor masks as instances
                tumor_inst, tumor_n = cc3d.connected_components(
                    tumor_bw, connectivity=26, return_N=True
                )
                
                # put the tumor instances back to gt_data_ori
                if tumor_n > 0:
                    gt_data_ori[tumor_inst > 0] = (
                        tumor_inst[tumor_inst > 0] + np.max(gt_data_ori) + 1
                    )
                    logger.info(f"找到 {tumor_n} 个肿瘤实例")
                else:
                    logger.info("没有找到肿瘤实例")
            except Exception as e:
                logger.error(f"处理肿瘤标签时出错: {str(e)}")
        
        # 记录3D小目标过滤前的状态
        non_zero_before_3d_filter = np.sum(gt_data_ori > 0)
        logger.info(f"3D过滤前非零像素数: {non_zero_before_3d_filter}")
        
        # exclude the objects with less than 1000 pixels in 3D
        try:
            gt_data_ori = cc3d.dust(
                gt_data_ori, threshold=voxel_num_thre3d, connectivity=26, in_place=True
            )
            
            # 记录3D过滤后的状态
            non_zero_after_3d_filter = np.sum(gt_data_ori > 0)
            logger.info(f"3D过滤后非零像素数: {non_zero_after_3d_filter}")
            
            if non_zero_before_3d_filter > 0 and non_zero_after_3d_filter == 0:
                logger.warning(f"警告: 3D过滤移除了所有标注，阈值为 {voxel_num_thre3d}")
        except Exception as e:
            logger.error(f"3D小目标过滤时出错: {str(e)}")
            failed_cases += 1
            continue
        
        # remove small objects with less than 100 pixels in 2D slices
        try:
            # 记录2D小目标过滤前的状态
            non_zero_before_2d_filter = non_zero_after_3d_filter
            slices_with_data = 0
            
            for slice_i in range(gt_data_ori.shape[0]):
                gt_i = gt_data_ori[slice_i, :, :]
                
                # 记录当前切片中的非零像素数
                non_zero_in_slice = np.sum(gt_i > 0)
                
                if non_zero_in_slice > 0:
                    slices_with_data += 1
                    if slices_with_data <= 5:  # 只记录前5个切片的详细信息，避免日志过长
                        logger.info(f"切片 {slice_i} 过滤前非零像素数: {non_zero_in_slice}")
                
                # remove small objects with less than 100 pixels
                gt_data_ori[slice_i, :, :] = cc3d.dust(
                    gt_i, threshold=voxel_num_thre2d, connectivity=8, in_place=True
                )
                
                # 记录过滤后的非零像素数
                if non_zero_in_slice > 0 and slices_with_data <= 5:
                    non_zero_after = np.sum(gt_data_ori[slice_i, :, :] > 0)
                    logger.info(f"切片 {slice_i} 过滤后非零像素数: {non_zero_after}")
                    if non_zero_in_slice > 0 and non_zero_after == 0:
                        logger.warning(f"警告: 切片 {slice_i} 的所有标注被2D过滤移除")
            
            # 记录2D过滤后的总体状态
            non_zero_after_2d_filter = np.sum(gt_data_ori > 0)
            logger.info(f"2D过滤后非零像素总数: {non_zero_after_2d_filter}")
            
            if non_zero_before_2d_filter > 0 and non_zero_after_2d_filter == 0:
                logger.warning(f"警告: 2D过滤移除了所有标注，阈值为 {voxel_num_thre2d}")
            
            logger.info(f"共有 {slices_with_data} 个切片包含数据")
        except Exception as e:
            logger.error(f"2D小目标过滤时出错: {str(e)}")
            failed_cases += 1
            continue
        
        # find non-zero slices
        z_index, _, _ = np.where(gt_data_ori > 0)
        z_index = np.unique(z_index)
        
        logger.info(f"找到 {len(z_index)} 个包含非零像素的切片")
        
        if len(z_index) > 0:
            logger.info("开始处理非零切片")
            
            # crop the ground truth with non-zero slices
            gt_roi = gt_data_ori[z_index, :, :]
            logger.info(f"裁剪后的标签形状: {gt_roi.shape}")
            
            # load image and preprocess
            try:
                logger.info(f"读取图像文件: {join(nii_path, image_name)}")
                img_sitk = sitk.ReadImage(join(nii_path, image_name))
                image_data = sitk.GetArrayFromImage(img_sitk)
                logger.info(f"图像形状: {image_data.shape}")
                
                # 记录图像值范围
                if len(image_data.flatten()) > 0:  # 确保有数据
                    min_val = np.min(image_data)
                    max_val = np.max(image_data)
                    mean_val = np.mean(image_data)
                    logger.info(f"图像值范围: 最小={min_val}, 最大={max_val}, 平均={mean_val:.2f}")
            except Exception as e:
                logger.error(f"读取或分析图像失败: {str(e)}")
                failed_cases += 1
                continue
            
            # nii preprocess start
            try:
                logger.info("开始预处理图像")
                
                if modality == "CT":
                    logger.info(f"使用CT窗位设置: 级别={WINDOW_LEVEL}, 窗宽={WINDOW_WIDTH}")
                    lower_bound = WINDOW_LEVEL - WINDOW_WIDTH / 2
                    upper_bound = WINDOW_LEVEL + WINDOW_WIDTH / 2
                    logger.info(f"裁剪范围: {lower_bound} 到 {upper_bound}")
                    
                    image_data_pre = np.clip(image_data, lower_bound, upper_bound)
                    image_data_pre = (
                        (image_data_pre - np.min(image_data_pre))
                        / (np.max(image_data_pre) - np.min(image_data_pre))
                        * 255.0
                    )
                else:
                    if len(image_data[image_data > 0]) > 0:  # 确保有非零数据
                        lower_bound = np.percentile(image_data[image_data > 0], 0.5)
                        upper_bound = np.percentile(image_data[image_data > 0], 99.5)
                        logger.info(f"使用百分位数裁剪: 0.5%={lower_bound}, 99.5%={upper_bound}")
                        
                        image_data_pre = np.clip(image_data, lower_bound, upper_bound)
                        image_data_pre = (
                            (image_data_pre - np.min(image_data_pre))
                            / (np.max(image_data_pre) - np.min(image_data_pre))
                            * 255.0
                        )
                        image_data_pre[image_data == 0] = 0
                    else:
                        logger.warning("图像中没有非零值，无法进行百分位数裁剪")
                        image_data_pre = np.zeros_like(image_data)

                image_data_pre = np.uint8(image_data_pre)
                img_roi = image_data_pre[z_index, :, :]
                logger.info(f"裁剪后的图像形状: {img_roi.shape}")
            except Exception as e:
                logger.error(f"图像预处理失败: {str(e)}")
                failed_cases += 1
                continue
            
            # 保存NPZ文件
            try:
                npz_path = join(npy_path, base_name+'.npz')
                logger.info(f"保存NPZ文件: {npz_path}")
                np.savez_compressed(npz_path, imgs=img_roi, gts=gt_roi, spacing=img_sitk.GetSpacing())
                logger.info(f"NPZ文件保存成功，包含 {img_roi.shape[0]} 个切片")
            except Exception as e:
                logger.error(f"保存NPZ文件失败: {str(e)}")
                failed_cases += 1
                continue
            
            # 创建该文件对应的子文件夹
            try:
                img_dir = join(npy_path, "imgs", base_name)
                gt_dir = join(npy_path, "gts", base_name)
                
                os.makedirs(img_dir, exist_ok=True)
                os.makedirs(gt_dir, exist_ok=True)
                logger.info(f"创建子目录: {img_dir}, {gt_dir}")
            except Exception as e:
                logger.error(f"创建子目录失败: {str(e)}")
                failed_cases += 1
                continue
            
            # 保存NII文件用于检查
            try:
                img_nii_path = join(npy_path, base_name + "_img.nii.gz")
                gt_nii_path = join(npy_path, base_name + "_gt.nii.gz")
                
                logger.info(f"保存NII检查文件: {img_nii_path}, {gt_nii_path}")
                
                img_roi_sitk = sitk.GetImageFromArray(img_roi)
                gt_roi_sitk = sitk.GetImageFromArray(gt_roi)
                
                sitk.WriteImage(img_roi_sitk, img_nii_path)
                sitk.WriteImage(gt_roi_sitk, gt_nii_path)
                
                logger.info("NII文件保存成功")
            except Exception as e:
                logger.error(f"保存NII文件失败: {str(e)}")
                # 继续处理，这不是致命错误
            
            # 保存每个切片为单独的NPY文件，从0开始命名
            try:
                logger.info(f"开始处理 {img_roi.shape[0]} 个切片")
                success_count = 0
                
                for i in range(img_roi.shape[0]):
                    try:
                        if i < 3 or i >= img_roi.shape[0] - 3:  # 只详细记录前3个和后3个切片
                            logger.info(f"处理切片 {i}/{img_roi.shape[0]}")
                        
                        # 处理图像切片
                        img_i = img_roi[i, :, :]
                        img_3c = np.repeat(img_i[:, :, None], 3, axis=-1)
                        
                        resize_img_skimg = transform.resize(
                            img_3c,
                            (image_size, image_size),
                            order=3,
                            preserve_range=True,
                            mode="constant",
                            anti_aliasing=True,
                        )
                        
                        resize_img_skimg_01 = (resize_img_skimg - resize_img_skimg.min()) / np.clip(
                            resize_img_skimg.max() - resize_img_skimg.min(), a_min=1e-8, a_max=None
                        )  # normalize to [0, 1], (H, W, 3)
                        
                        # 处理标签切片
                        gt_i = gt_roi[i, :, :]
                        resize_gt_skimg = transform.resize(
                            gt_i,
                            (image_size, image_size),
                            order=0,
                            preserve_range=True,
                            mode="constant",
                            anti_aliasing=False,
                        )
                        resize_gt_skimg = np.uint8(resize_gt_skimg)
                        
                        # 检查形状
                        if resize_img_skimg_01.shape[:2] != resize_gt_skimg.shape:
                            logger.warning(f"切片 {i} 的图像和标签形状不匹配: {resize_img_skimg_01.shape[:2]} vs {resize_gt_skimg.shape}")
                            continue
                        
                        # 保存NPY文件
                        img_save_path = join(npy_path, "imgs", base_name, str(i).zfill(3) + ".npy")
                        gt_save_path = join(npy_path, "gts", base_name, str(i).zfill(3) + ".npy")
                        
                        np.save(img_save_path, resize_img_skimg_01)
                        np.save(gt_save_path, resize_gt_skimg)
                        
                        success_count += 1
                    except Exception as e:
                        logger.error(f"处理切片 {i} 时出错: {str(e)}")
                
                logger.info(f"成功保存 {success_count}/{img_roi.shape[0]} 个切片")
            except Exception as e:
                logger.error(f"处理切片时出错: {str(e)}")
                failed_cases += 1
                continue
            
            processed_cases += 1
            logger.info(f"案例 {name} 处理完成")
        else:
            logger.warning(f"案例 {name} 不包含任何有效标注区域，跳过处理")
            # 诊断原因
            logger.warning("可能的原因:")
            
            if non_zero_count == 0:
                logger.warning("原始标签文件中没有非零像素")
            elif non_zero_after_removal == 0:
                logger.warning(f"所有非零像素都被移除标签列表 {remove_label_ids} 排除")
            elif non_zero_before_3d_filter > 0 and non_zero_after_3d_filter == 0:
                logger.warning(f"所有标注区域都小于3D体素阈值 {voxel_num_thre3d}")
            elif non_zero_before_2d_filter > 0 and non_zero_after_2d_filter == 0:
                logger.warning(f"所有标注区域都小于2D像素阈值 {voxel_num_thre2d}")
            
            empty_cases += 1
    
    except Exception as e:
        logger.error(f"处理案例 {name} 时出现未处理的异常: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        failed_cases += 1
    
    case_time = time.time() - case_start_time
    logger.info(f"案例 {name} 处理耗时: {case_time:.2f} 秒")
    logger.info("-"*50)

# 打印总结
total_time = time.time() - start_time
logger.info("="*50)
logger.info("处理完成，统计信息：")
logger.info(f"总案例数: {total_cases}")
logger.info(f"成功处理: {processed_cases}")
logger.info(f"空案例(无标注): {empty_cases}")
logger.info(f"处理失败: {failed_cases}")
logger.info(f"总耗时: {total_time:.2f} 秒")
logger.info(f"日志文件: {log_file}")

if empty_cases > 0:
    logger.info("\n空案例原因可能是:")
    logger.info("1. 原始标签文件为空")
    logger.info("2. 所有标签值都在移除列表中")
    logger.info("3. 标注区域太小，被过滤阈值移除")
    logger.info("\n解决方案:")
    logger.info(f"1. 检查标签文件是否有有效数据")
    logger.info(f"2. 修改移除标签列表 (当前为: {remove_label_ids})")
    logger.info(f"3. 减小3D体素阈值 (当前为: {voxel_num_thre3d}) 或2D像素阈值 (当前为: {voxel_num_thre2d})")

logger.info("="*50)