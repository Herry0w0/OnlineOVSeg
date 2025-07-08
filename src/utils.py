"""
Core utility functions and base classes
"""
import torch
import torch.nn as nn
import numpy as np
import yaml
from typing import Dict, Any, Optional, Tuple, List
import logging
import os

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                   epoch: int, loss: float, save_path: str):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, save_path)

def load_checkpoint(checkpoint_path: str, model: nn.Module, 
                   optimizer: Optional[torch.optim.Optimizer] = None):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss']

def compute_centroid(points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute centroid of masked points"""
    masked_points = points[mask.bool()]
    if len(masked_points) == 0:
        return torch.zeros_like(points[0])
    return masked_points.mean(dim=0)

def fps_sampling(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Farthest Point Sampling"""
    B, N, C = points.shape
    centroids = torch.zeros(B, num_samples, dtype=torch.long).to(points.device)
    distance = torch.ones(B, N).to(points.device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(points.device)
    
    for i in range(num_samples):
        centroids[:, i] = farthest
        centroid = points[torch.arange(B), farthest, :].view(B, 1, C)
        dist = torch.sum((points - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    
    return centroids
