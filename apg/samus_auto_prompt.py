import torch
from torch import nn
from einops import rearrange
from typing import Type, Tuple
import torch.nn.functional as F

# 自定义softmax函数，提高数值稳定性
# 通过减去最大值避免指数运算溢出
def softmax_one(x, dim=-1):
    x = x - x.max(dim=dim, keepdim=True)[0]
    return torch.softmax(x, dim=dim)

# Vision Transformer注意力机制实现
class vitAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads  # 注意力头的总维度
        # 判断是否需要投影输出层
        project_out = not (heads == 1 and dim_head == dim)
        
        self.heads = heads
        self.scale = dim_head ** -0.5  # 缩放因子，防止点积过大
        self.attend = nn.Softmax(dim=-1)
        
        # 查询、键、值的线性变换层
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        
        # 输出投影层
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, qx, kx):
        # 生成查询向量
        q = self.to_q(qx)
        # 重塑查询向量形状: [batch, heads, seq_len, dim_per_head]
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        
        # 生成键和值向量
        kv = self.to_kv(kx).chunk(2, dim=-1)
        # 重塑键和值向量形状
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv)
        
        # 计算注意力分数
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # 应用softmax获取注意力权重
        attn = softmax_one(dots, dim=-1)
        
        # 应用注意力权重到值向量
        out = torch.matmul(attn, v)
        # 重新排列输出形状
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        # 返回经过输出投影层的结果
        return self.to_out(out)

# 预归一化模块 - 单输入
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # 层归一化
        self.fn = fn  # 被包装的函数

    def forward(self, x, **kwargs):
        # 先归一化再执行函数
        return self.fn(self.norm(x), **kwargs)

# 预归一化模块 - 双输入
class PreNorm2in(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        # 两个独立的归一化层
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fn = fn  # 被包装的函数

    def forward(self, x1, x2, **kwargs):
        # 分别对两个输入进行归一化后再执行函数
        return self.fn(self.norm1(x1), self.norm2(x2), **kwargs)

# 前馈网络
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        # 两层线性变换，中间使用ReLU激活
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# 交叉注意力Transformer
class CrossTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        
        # 构建多层Transformer结构
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # 交叉注意力: x1关注x2
                PreNorm2in(dim, vitAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                # 交叉注意力: x2关注x1
                PreNorm2in(dim, vitAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                # x1的前馈网络
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                # x2的前馈网络
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x1, x2):
        # 逐层处理
        for attn1, attn2, ff1, ff2 in self.layers:
            # 双向交叉注意力
            ax1, ax2 = attn1(x1, x2), attn2(x2, x1)
            # 残差连接
            x1, x2 = ax1 + x1, ax2 + x2
            # 前馈网络和残差连接
            x1 = ff1(x1) + x1
            x2 = ff2(x2) + x2
        return x1, x2

# Prompt嵌入生成器
class Prompt_Embedding_Generator(nn.Module):
    def __init__(
        self,
        out_dim: int = 256,      # 输出维度
        base_dim: int = 48,      # 基础维度
        num_heads: int = 8,      # 注意力头数
        activation: Type[nn.Module] = nn.GELU,  # 激活函数类型
    ) -> None:
        super().__init__()
        self.embed_dim = out_dim
        self.base_dim = base_dim
        self.num_heads = num_heads
        self.scale = (out_dim//self.num_heads)**-0.5
        
        # 可学习的对象token参数
        self.object_token = nn.Parameter(torch.randn(1, 50, self.embed_dim))
        
        # 交叉注意力模块
        self.cross_token_token = CrossTransformer(dim=self.embed_dim, depth=2, heads=8, dim_head=64)
        self.token_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
        # 图像- token交叉注意力模块
        self.cross_image_token = CrossTransformer(dim=self.embed_dim, depth=2, heads=8, dim_head=64)

    def forward(self,
        img_embedding: torch.Tensor,    # 图像嵌入 [B, C, H, W]
        output_token: torch.Tensor,     # 输出token [B, N, D]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = img_embedding.shape
        # 重排图像嵌入为序列形式
        img_embedding = rearrange(img_embedding, 'b c h w -> b (h w) c')
        
        # 通过token-token交叉注意力更新object_token和output_token
        object_token, new_output_token = self.cross_token_token(self.object_token, output_token)
        # 投影和残差连接
        object_token = self.token_proj(object_token) + self.object_token
        new_output_token = self.token_proj(new_output_token) + output_token
        
        # 拼接tokens
        tokens = torch.cat([object_token, output_token], dim=1)  # [b 6 d]
        
        # 通过image-token交叉注意力更新图像嵌入和tokens
        new_img_embedding, tokens = self.cross_image_token(img_embedding, tokens) 
        
        # 重排图像嵌入回原始形状
        new_img_embedding = rearrange(new_img_embedding, 'b (h w) c -> b c h w', h=h)
        
        return new_img_embedding, object_token, new_output_token 