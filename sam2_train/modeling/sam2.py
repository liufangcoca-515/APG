from typing import Any, Dict, List, Tuple, Union, Optional

import torch
from torch.nn import functional as F

from .sam2_base import SAM2Base
from .backbones.build_backbone import build_backbone
from .mem_attention import build_memory_attention
from .mem_encoder import build_memory_encoder


def build_sam2_model(
    model_type: str,
    checkpoint: Optional[str] = None,
    use_high_res_features=False,
    multimask_output=False,
    use_samus_style_prompt=False,  # 添加对SAMUS风格prompt的支持
    **kwargs,
):
    """
    构建SAM2模型
    
    Args:
        model_type (str): 模型类型，如 "vit_h", "vit_l", "vit_b"
        checkpoint (str, optional): 预训练检查点路径
        use_high_res_features (bool): 是否使用高分辨率特征
        multimask_output (bool): 是否输出多个掩码候选
        use_samus_style_prompt (bool): 是否使用SAMUS风格的prompt生成器
        **kwargs: 其他参数
        
    Returns:
        SAM2Base: SAM2模型实例
    """
    # 构建图像编码器
    image_encoder = build_backbone(
        backbone_type=model_type, 
        use_high_res_features=use_high_res_features
    )
    
    # 构建记忆注意力模块
    memory_attention = build_memory_attention(
        image_encoder.out_dim, 
        image_encoder.out_dim
    )
    
    # 构建记忆编码器
    memory_encoder = build_memory_encoder(image_encoder.out_dim)
    
    # 构建SAM2模型
    model = SAM2Base(
        image_encoder=image_encoder,
        memory_attention=memory_attention,
        memory_encoder=memory_encoder,
        use_high_res_features_in_sam=use_high_res_features,
        multimask_output_in_sam=multimask_output,
        use_samus_style_prompt=use_samus_style_prompt,  # 添加SAMUS风格prompt
        **kwargs,
    )
    
    # 加载预训练权重（如果提供）
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    
    return model 