"""
ScanNet dataset loader for multi-frame point clouds and images
"""
import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
import pickle
import json
from PIL import Image
import logging
import open3d as o3d

logger = logging.getLogger(__name__)

class ScanNetMultiFrameDataset(Dataset):
    """
    ScanNet dataset loader that loads adjacent frames with point clouds,
    RGB images, and instance segmentation masks
    """
    
    def __init__(self, 
                 data_root: str = "/media/ssd/jiangxirui/projects/2/data/ScanNetV2",
                 split: str = "train",
                 num_frames: int = 5,
                 step_size: int = 25,
                 max_points: int = 50000,
                 image_size: Tuple[int, int] = (1296, 968)):
        """
        Args:
            data_root: Path to ScanNet dataset
            split: Dataset split (train/val/test)
            num_frames: Number of adjacent frames to load
            max_points: Maximum number of points per frame
            image_size: Target image size (width, height)
        """
        self.data_root = data_root
        self.split = split
        self.num_frames = num_frames
        self.step_size = step_size
        self.max_points = max_points
        self.image_size = image_size
        
        # Load scene list
        self.scenes = self._load_scene_list()
        
        # Build frame index
        self.frame_index = self._build_frame_index()
        
        logger.info(f"Loaded {len(self.frame_index)} frame sequences for {split} split")
    
    def _load_scene_list(self) -> List[str]:
        """Load list of scenes for the split"""
        split_file = os.path.join(self.data_root, f"meta/scannetv2_{self.split}.txt")
        
        with open(split_file, 'r') as f:
            scenes = [line.strip() for line in f.readlines()]
        return scenes
    
    def _build_frame_index(self) -> List[Dict]:
        """Build index of valid frame sequences"""
        frame_index = []
        
        for scene_id in self.scenes:
            scene_path = os.path.join(self.data_root, scene_id)
            if not os.path.exists(scene_path):
                continue
                
            # Get available frame indices
            frame_files = sorted([f for f in os.listdir(scene_path) 
                                if f.endswith('.jpg')])
            frame_indices = [int(f.split('.')[0]) for f in frame_files]
            
            # Create sequences of adjacent frames
            for start_idx in range(len(frame_indices) - (self.num_frames - 1) * self.step_size):
                if start_idx % self.step_size == 0:
                    frame_sequence = [frame_indices[start_idx + i * self.step_size] for i in range(self.num_frames)]
                    frame_index.append({
                        'scene_id': scene_id,
                        'frame_indices': frame_sequence,
                        'scene_path': scene_path
                    })
        
        return frame_index
    
    def __len__(self) -> int:
        return len(self.frame_index)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            Dictionary containing:
            - point_clouds: [num_frames, max_points, 6] (xyz + rgb)
            - images: [num_frames, 3, H, W]
            - instance_masks: [num_frames, H, W]
            - frame_info: metadata about frames
            - scene_point_cloud: Complete scene point cloud
            - visibility_matrix: Point visibility across frames
            - point_indices: Frame-to-scene point mapping
        """
        sample = self.frame_index[idx]
        scene_id = sample['scene_id']
        frame_indices = sample['frame_indices']
        scene_path = sample['scene_path']
        
        # Load complete scene point cloud first
        scene_point_cloud = self._load_scene_point_cloud(scene_path)
        
        # Load frame data
        point_clouds = []
        images = []
        instance_masks = []
        point_indices = []
        
        for frame_idx in frame_indices:
            # Load RGB image
            image = self._load_image(scene_path, frame_idx)
            images.append(image)
            
            # Load instance mask
            mask = self._load_instance_mask(scene_path, frame_idx)
            instance_masks.append(mask)
            
            # Load frame point cloud and get indices
            frame_pc, frame_indices_map = self._load_frame_point_cloud(
                scene_path, frame_idx, scene_point_cloud)
            point_clouds.append(frame_pc)
            point_indices.append(frame_indices_map)
        
        # Compute visibility matrix
        visibility_matrix = self._compute_visibility_matrix(point_indices, len(scene_point_cloud))
        
        return {
            'point_clouds': torch.stack(point_clouds),
            'images': torch.stack(images),
            'instance_masks': torch.stack(instance_masks),
            'scene_point_cloud': scene_point_cloud,
            'visibility_matrix': visibility_matrix,
            'point_indices': point_indices,
            'scene_id': scene_id,
            'frame_indices': frame_indices
        }
    
    def _load_scene_point_cloud(self, scene_path: str) -> torch.Tensor:
        """Load complete scene point cloud"""

        # Try to load cached scene point cloud
        scene_pc_path = os.path.join(scene_path, "scene_point_cloud.pth")
        if os.path.exists(scene_pc_path):
            return torch.load(scene_pc_path)
        
        # Otherwise, aggregate from all frames (simplified)
        ply_path = os.path.join(scene_path, f"{os.path.basename(scene_path)}_vh_clean_2.ply")

        pcd = o3d.io.read_point_cloud(ply_path)          
        # Extract coordinates
        coords = np.asarray(pcd.points)  # (N, 3)
        # Extract colors
        colors = np.asarray(pcd.colors)  # (N, 3), normalized [0, 1]
        # Combine into Nx6 tensor
        scene_pc = torch.tensor(np.concatenate([coords, colors], axis=1), dtype=torch.float32)

        # Cache for future use
        torch.save(scene_pc, scene_pc_path)
        return scene_pc
    
    def _load_image(self, scene_path: str, frame_idx: int) -> torch.Tensor:
        """Load RGB image"""
        image_path = os.path.join(scene_path, f"{frame_idx}.jpg")
        
        image = Image.open(image_path).convert('RGB')
        image = image.resize(self.image_size)
        image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0
        return image
    
    def _load_instance_mask(self, scene_path: str, frame_idx: int) -> torch.Tensor:
        """Load instance segmentation mask"""
        # ScanNet instance masks are typically stored as PNG files
        mask_path = os.path.join(scene_path, "2D_instance_masks", f"{frame_idx}.png")
        
        mask = cv2.imread(mask_path, -1)
        # mask = mask.resize(self.image_size, Image.NEAREST)
        mask = torch.tensor(np.array(mask), dtype=torch.long)
        return mask
    
    def _load_frame_point_cloud(self, scene_path: str, frame_idx: int, 
                               scene_point_cloud: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load frame point cloud and compute mapping to scene point cloud"""
        # For simplicity, we'll sample from scene point cloud
        # In practice, you'd use camera poses and depth maps to get actual frame points
        
        # Sample random subset of scene points (simplified approach)
        num_scene_points = len(scene_point_cloud)
        frame_size = min(self.max_points, num_scene_points // 2)
        
        # Random sampling for now - in practice use camera projection
        indices = torch.randperm(num_scene_points)[:frame_size]
        frame_pc = scene_point_cloud[indices]
        
        # Pad if necessary
        if len(frame_pc) < self.max_points:
            padding = torch.zeros(self.max_points - len(frame_pc), 6)
            frame_pc = torch.cat([frame_pc, padding], dim=0)
            # Extend indices with -1 for padding
            padding_indices = torch.full((self.max_points - len(indices),), -1, dtype=torch.long)
            indices = torch.cat([indices, padding_indices])
        
        return frame_pc, indices
    
    def _compute_visibility_matrix(self, point_indices: List[torch.Tensor], 
                                  num_scene_points: int) -> torch.Tensor:
        """Compute visibility matrix showing which scene points are visible in which frames"""
        visibility = torch.zeros(num_scene_points, self.num_frames, dtype=torch.bool)
        
        for frame_idx, indices in enumerate(point_indices):
            valid_indices = indices[indices >= 0]  # Remove padding indices
            visibility[valid_indices, frame_idx] = True
        
        return visibility