"""
Simplified PointTransformerV3 implementation
Based on the PointTransformerV3 architecture but without mmcv dependencies
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math

class PointTransformerBlock(nn.Module):
    """Single Point Transformer block"""
    
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Multi-head attention
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.1)
        
        # Position encoding
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        
        # Feed forward
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(0.1)
        )
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, C] point features
            pos: [B, N, 3] point positions
        Returns:
            [B, N, C] updated point features
        """
        B, N, C = x.shape
        
        # Position encoding
        pos_enc = self.pos_mlp(pos)
        
        # Self attention with position encoding
        shortcut = x
        x = self.norm1(x)
        
        qkv = self.qkv(x + pos_enc).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = shortcut + x
        
        # Feed forward
        x = x + self.mlp(self.norm2(x))
        
        return x

class PointTransformerV3(nn.Module):
    """Simplified PointTransformerV3 backbone"""
    
    def __init__(self, 
                 input_dim: int = 6,  # xyz + rgb
                 output_dim: int = 512,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 embed_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            PointTransformerBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, output_dim)
        )
        
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: [B, N, 6] point cloud with xyz + rgb
        Returns:
            [B, N, output_dim] point features
        """
        B, N, _ = points.shape
        
        # Split coordinates and features
        pos = points[:, :, :3]  # xyz coordinates
        feat = points  # full features including xyz and rgb
        
        # Input projection
        x = self.input_proj(feat)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, pos)
        
        x = self.norm(x)
        
        # Output projection
        features = self.output_proj(x)
        
        return features

class FeatureAdapter(nn.Module):
    """Adapter to align PTv3 features with CLIP text features"""
    
    def __init__(self, 
                 input_dim: int = 512,
                 output_dim: int = 512,
                 hidden_dims: list = [256, 512]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.adapter = nn.Sequential(*layers)
        
        # L2 normalization for similarity computation
        self.normalize = nn.functional.normalize
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, N, input_dim] point features
        Returns:
            [B, N, output_dim] adapted features
        """
        adapted = self.adapter(features)
        # L2 normalize for cosine similarity
        adapted = self.normalize(adapted, p=2, dim=-1)
        return adapted
