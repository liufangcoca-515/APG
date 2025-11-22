#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from PIL import Image
import argparse
from tqdm import tqdm
import shutil

def npy_to_jpg(npy_folder, jpg_folder):
    """
    将图像的NPY格式转换为JPG格式
    
    参数:
    npy_folder: NPY文件所在文件夹路径
    jpg_folder: 要保存JPG文件的文件夹路径
    """
    print(f"正在处理文件夹: {npy_folder}")
    os.makedirs(jpg_folder, exist_ok=True)
    
    # 获取所有子文件夹
    subdirs = [d for d in os.listdir(npy_folder) if os.path.isdir(os.path.join(npy_folder, d))]
    print(f"找到 {len(subdirs)} 个子文件夹")
    
    if subdirs:
        # 处理每个子文件夹
        for subdir in subdirs:
            print(f"处理子文件夹: {subdir}")
            npy_subdir = os.path.join(npy_folder, subdir)
            jpg_subdir = os.path.join(jpg_folder, subdir)
            os.makedirs(jpg_subdir, exist_ok=True)
            
            # 获取子文件夹中的所有npy文件
            npy_files = [f for f in os.listdir(npy_subdir) if f.endswith('.npy')]
            print(f"在子文件夹 {subdir} 中找到 {len(npy_files)} 个npy文件")
            
            for npy_file in tqdm(npy_files, desc=f"转换 {subdir} 中的图像文件"):
                # 加载npy文件
                npy_path = os.path.join(npy_subdir, npy_file)
                print(f"正在处理文件: {npy_path}")
                data = np.load(npy_path)
                print(f"数据形状: {data.shape}, 数据类型: {data.dtype}")
                
                # 文件名(不含扩展名)
                base_name = os.path.splitext(npy_file)[0]
                
                # 处理图像数据
                if data.ndim == 3 and data.shape[2] == 3:
                    print("处理RGB图像")
                    # 如果已经是RGB格式，确保值在0-255范围内
                    if data.dtype == np.float32 or data.dtype == np.float64:
                        if np.max(data) <= 1.0:
                            data = (data * 255).astype(np.uint8)
                        else:
                            data = data.astype(np.uint8)
                    img = Image.fromarray(data)
                elif data.ndim == 2:
                    print("处理灰度图像")
                    # 如果是灰度图，转换为RGB
                    if data.dtype == np.float32 or data.dtype == np.float64:
                        if np.max(data) <= 1.0:
                            data = (data * 255).astype(np.uint8)
                        else:
                            data = data.astype(np.uint8)
                    img = Image.fromarray(data).convert('RGB')
                else:
                    print(f"警告：无法处理形状为{data.shape}的数据")
                    raise ValueError(f"无法处理形状为{data.shape}的数据")
                
                # 保存为JPG
                jpg_path = os.path.join(jpg_subdir, f"{base_name}.jpg")
                print(f"保存到: {jpg_path}")
                img.save(jpg_path, quality=95)
    else:
        # 如果没有子文件夹，直接处理当前文件夹的npy文件
        npy_files = [f for f in os.listdir(npy_folder) if f.endswith('.npy')]
        print(f"找到 {len(npy_files)} 个npy文件")
        
        for npy_file in tqdm(npy_files, desc="转换图像文件"):
            # 加载npy文件
            npy_path = os.path.join(npy_folder, npy_file)
            print(f"正在处理文件: {npy_path}")
            data = np.load(npy_path)
            print(f"数据形状: {data.shape}, 数据类型: {data.dtype}")
            
            # 文件名(不含扩展名)
            base_name = os.path.splitext(npy_file)[0]
            
            # 处理图像数据
            if data.ndim == 3 and data.shape[2] == 3:
                print("处理RGB图像")
                # 如果已经是RGB格式，确保值在0-255范围内
                if data.dtype == np.float32 or data.dtype == np.float64:
                    if np.max(data) <= 1.0:
                        data = (data * 255).astype(np.uint8)
                    else:
                        data = data.astype(np.uint8)
                img = Image.fromarray(data)
            elif data.ndim == 2:
                print("处理灰度图像")
                # 如果是灰度图，转换为RGB
                if data.dtype == np.float32 or data.dtype == np.float64:
                    if np.max(data) <= 1.0:
                        data = (data * 255).astype(np.uint8)
                    else:
                        data = data.astype(np.uint8)
                img = Image.fromarray(data).convert('RGB')
            else:
                print(f"警告：无法处理形状为{data.shape}的数据")
                raise ValueError(f"无法处理形状为{data.shape}的数据")
            
            # 保存为JPG
            jpg_path = os.path.join(jpg_folder, f"{base_name}.jpg")
            print(f"保存到: {jpg_path}")
            img.save(jpg_path, quality=95)

def copy_label_npy(npy_folder, target_folder):
    """
    直接复制标签的NPY文件到目标文件夹
    
    参数:
    npy_folder: 标签NPY文件所在文件夹路径
    target_folder: 要保存标签NPY文件的目标文件夹路径
    """
    os.makedirs(target_folder, exist_ok=True)
    
    # 获取所有npy文件
    npy_files = [f for f in os.listdir(npy_folder) if f.endswith('.npy')]
    
    for npy_file in tqdm(npy_files, desc="复制标签文件"):
        # 源文件和目标文件路径
        src_path = os.path.join(npy_folder, npy_file)
        dst_path = os.path.join(target_folder, npy_file)
        
        # 复制文件
        shutil.copy2(src_path, dst_path)

def process_folder_structure(npy_base_folder, output_base_folder):
    """处理整个文件夹结构，图像转为jpg，标签保持npy格式"""
    print(f"开始处理文件夹结构...")
    print(f"输入文件夹: {npy_base_folder}")
    print(f"输出文件夹: {output_base_folder}")
    
    # 处理imgs文件夹 - 转换为jpg
    imgs_npy_folder = os.path.join(npy_base_folder, "imgs")
    imgs_jpg_folder = os.path.join(output_base_folder, "imgs")
    
    if os.path.exists(imgs_npy_folder):
        print(f"找到图像文件夹: {imgs_npy_folder}")
        # 获取所有子文件夹
        subdirs = [d for d in os.listdir(imgs_npy_folder) if os.path.isdir(os.path.join(imgs_npy_folder, d))]
        print(f"找到 {len(subdirs)} 个子文件夹")
        
        if subdirs:
            # 如果有子文件夹，处理每个子文件夹
            for subdir in subdirs:
                npy_subdir = os.path.join(imgs_npy_folder, subdir)
                jpg_subdir = os.path.join(imgs_jpg_folder, subdir)
                
                print(f"处理图像子文件夹: {subdir}")
                npy_to_jpg(npy_subdir, jpg_subdir)
        else:
            # 如果没有子文件夹，直接处理
            print("没有找到子文件夹，直接处理当前文件夹")
            npy_to_jpg(imgs_npy_folder, imgs_jpg_folder)
    else:
        print(f"警告：图像文件夹不存在: {imgs_npy_folder}")
    
    # 处理gts文件夹 - 保持npy格式
    gts_npy_folder = os.path.join(npy_base_folder, "gts")
    gts_target_folder = os.path.join(output_base_folder, "gts")
    
    if os.path.exists(gts_npy_folder):
        print("处理标签文件夹...")
        # 获取所有子文件夹
        subdirs = [d for d in os.listdir(gts_npy_folder) if os.path.isdir(os.path.join(gts_npy_folder, d))]
        
        if subdirs:
            # 如果有子文件夹，处理每个子文件夹
            for subdir in subdirs:
                npy_subdir = os.path.join(gts_npy_folder, subdir)
                target_subdir = os.path.join(gts_target_folder, subdir)
                
                print(f"处理标签子文件夹: {subdir}")
                copy_label_npy(npy_subdir, target_subdir)
        else:
            # 如果没有子文件夹，直接处理
            copy_label_npy(gts_npy_folder, gts_target_folder)

def main():
    parser = argparse.ArgumentParser(description='将图像的NPY格式转换为JPG格式，保持标签为NPY格式')
    parser.add_argument('--npy_folder', required=True, help='NPY文件所在的文件夹路径')
    parser.add_argument('--output_folder', required=True, help='输出文件夹路径')
    parser.add_argument('--process_structure', action='store_true', help='是否处理整个文件夹结构(包含imgs和gts子文件夹)')
    parser.add_argument('--only_images', action='store_true', help='仅处理图像文件，不处理标签')
    
    args = parser.parse_args()
    
    print("启动参数：")
    print(f"NPY文件夹: {args.npy_folder}")
    print(f"输出文件夹: {args.output_folder}")
    print(f"处理结构: {args.process_structure}")
    print(f"仅处理图像: {args.only_images}")
    
    if args.process_structure:
        process_folder_structure(args.npy_folder, args.output_folder)
    elif args.only_images:
        npy_to_jpg(args.npy_folder, args.output_folder)
    else:
        print("请使用 --process_structure 或 --only_images 参数指定处理方式")
    
    print("处理完成!")

if __name__ == "__main__":
    main() 