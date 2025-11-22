""" Dataloader for the BTCV dataset
    Yunli Qi
"""
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from func_3d.utils import random_click, generate_bbox


class BTCV(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', seed=None, variation=0):

        # Set the data list for training
        self.name_list = os.listdir(os.path.join(data_path, mode, 'image'))
        
        # Set the basic information of the dataset
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk
        self.seed = seed
        self.variation = variation
        if mode == 'Training':
            self.video_length = args.video_length
        else:
            self.video_length = None

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        point_label = 1
        newsize = (self.img_size, self.img_size)

        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, self.mode, 'image', name)
        mask_path = os.path.join(self.data_path, self.mode, 'mask', name)
        
        # 打印样本文件夹中有多少图片文件
        img_files = os.listdir(img_path)
        # print(f"样本 {name} 中的图片文件数量: {len(img_files)}")
        # print(f"图片文件列表: {img_files}")
        
        data_seg_3d_shape = np.load(mask_path + '/0.npy').shape
        num_frame = len(os.listdir(mask_path))
        # print(f"样本 {name} 中的mask文件数量: {num_frame}")
        
        data_seg_3d = np.zeros(data_seg_3d_shape + (num_frame,))
        for i in range(num_frame):
            data_seg_3d[..., i] = np.load(os.path.join(mask_path, f'{i}.npy'))
        for i in range(data_seg_3d.shape[-1]):
            if np.sum(data_seg_3d[..., i]) > 0:
                data_seg_3d = data_seg_3d[..., i:]
                break
        starting_frame_nonzero = i
        # print(f"样本 {name} 的第一个非零帧索引: {starting_frame_nonzero}")
        
        for j in reversed(range(data_seg_3d.shape[-1])):
            if np.sum(data_seg_3d[..., j]) > 0:
                data_seg_3d = data_seg_3d[..., :j+1]
                break
        num_frame = data_seg_3d.shape[-1]
        # print(f"样本 {name} 处理后的有效帧数量: {num_frame}")
        
        if self.video_length is None:
            video_length = num_frame
        else:
            video_length = self.video_length
        if num_frame > video_length and self.mode == 'Training':
            starting_frame = np.random.randint(0, num_frame - video_length + 1)
        else:
            starting_frame = 0
        # print(f"样本 {name} 的起始帧索引: {starting_frame}, 视频长度: {video_length}")
        
        img_tensor = torch.zeros(video_length, 3, self.img_size, self.img_size)
        mask_dict = {}
        point_label_dict = {}
        pt_dict = {}
        bbox_dict = {}

        # 记录实际加载的帧数和对象数
        loaded_frames = []
        all_obj_ids = set()

        for frame_index in range(starting_frame, starting_frame + video_length):
            img_file = f'{frame_index + starting_frame_nonzero}.jpg'
            # print(f"样本 {name} 加载图片: {img_file}")
            
            img = Image.open(os.path.join(img_path, img_file)).convert('RGB')
            mask = data_seg_3d[..., frame_index]
            # mask = np.rot90(mask)
            obj_list = np.unique(mask[mask > 0])
            # print(f"样本 {name} 帧 {frame_index} 中的对象ID: {obj_list}")
            all_obj_ids.update(obj_list)
            
            diff_obj_mask_dict = {}
            if self.prompt == 'bbox':
                diff_obj_bbox_dict = {}
            elif self.prompt == 'click':
                diff_obj_pt_dict = {}
                diff_obj_point_label_dict = {}
            else:
                raise ValueError('Prompt not recognized')
            for obj in obj_list:
                obj_mask = mask == obj
                # if self.transform_msk:
                obj_mask = Image.fromarray(obj_mask)
                obj_mask = obj_mask.resize(newsize)
                obj_mask = torch.tensor(np.array(obj_mask)).unsqueeze(0).int()
                    # obj_mask = self.transform_msk(obj_mask).int()
                diff_obj_mask_dict[obj] = obj_mask

                if self.prompt == 'click':
                    diff_obj_point_label_dict[obj], diff_obj_pt_dict[obj] = random_click(np.array(obj_mask.squeeze(0)), point_label, seed=None)
                if self.prompt == 'bbox':
                    diff_obj_bbox_dict[obj] = generate_bbox(np.array(obj_mask.squeeze(0)), variation=self.variation, seed=self.seed)
            # if self.transform:
                # state = torch.get_rng_state()
                # img = self.transform(img)
                # torch.set_rng_state(state)
            img = img.resize(newsize)
            img = torch.tensor(np.array(img)).permute(2, 0, 1)

            img_tensor[frame_index - starting_frame, :, :, :] = img
            mask_dict[frame_index - starting_frame] = diff_obj_mask_dict
            if self.prompt == 'bbox':
                bbox_dict[frame_index - starting_frame] = diff_obj_bbox_dict
            elif self.prompt == 'click':
                pt_dict[frame_index - starting_frame] = diff_obj_pt_dict
                point_label_dict[frame_index - starting_frame] = diff_obj_point_label_dict
            
            loaded_frames.append(frame_index - starting_frame)

        # print(f"样本 {name} 最终加载的帧索引: {loaded_frames}")
        # print(f"样本 {name} 所有对象ID: {all_obj_ids}")
        # print(f"样本 {name} 总共加载了 {len(loaded_frames)} 帧，包含 {len(all_obj_ids)} 个对象")

        image_meta_dict = {'filename_or_obj':name}
        if self.prompt == 'bbox':
            return {
                'image':img_tensor,
                'label': mask_dict,
                'bbox': bbox_dict,
                'image_meta_dict':image_meta_dict,
            }
        elif self.prompt == 'click':
            return {
                'image':img_tensor,
                'label': mask_dict,
                'p_label':point_label_dict,
                'pt':pt_dict,
                'image_meta_dict':image_meta_dict,
            }