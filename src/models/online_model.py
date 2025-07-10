"""
Main model that combines PointTransformerV3, adapter, and CLIP
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import logging

from .pointtransformer import PointTransformerV3, FeatureAdapter
from .clip_encoder import CLIPTextEncoder

logger = logging.getLogger(__name__)

class OnlineInstanceSegmentationModel(nn.Module):
    """
    Main model for online 3D instance segmentation with multi-modal supervision
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Point cloud backbone
        self.ptv3 = PointTransformerV3(
            input_dim=config['ptv3']['input_dim'],
            output_dim=config['ptv3']['output_dim'],
            num_layers=config['ptv3']['num_layers'],
            num_heads=config['ptv3']['num_heads']
        )
        
        # Feature adapter for CLIP alignment
        self.adapter = FeatureAdapter(
            input_dim=config['adapter']['input_dim'],
            output_dim=config['adapter']['output_dim'],
            hidden_dims=config['adapter']['hidden_dims']
        )
        
        # CLIP text encoder
        self.clip_encoder = CLIPTextEncoder(config['clip']['model_name'])
        
        # LLaVA text generator (mock for now)
        self.llava_generator = LLaVATextGenerator()
        
        # Feature memory for cross-frame consistency
        self.feature_memory = {}
        self.consistency_threshold = 0.8
        
        logger.info("Initialized OnlineInstanceSegmentationModel")
    
    def forward(self, batch: Dict) -> Dict:
        """
        Forward pass for training
        
        Args:
            batch: Dictionary containing:
                - point_clouds: [B, num_frames, N, 6]
                - images: [B, num_frames, 3, H, W]
                - instance_masks: [B, num_frames, H, W]
                - scene_point_cloud: [B, total_points, 6]
                - visibility_matrix: [B, total_points, num_frames]
                - point_indices: List of [B, N] tensors
        Returns:
            Dictionary containing features and predictions
        """
        B, num_frames, N, _ = batch['point_clouds'].shape
        
        # Process each frame through PTv3
        all_features = []
        adapted_features = []
        
        for frame_idx in range(num_frames):
            frame_points = batch['point_clouds'][:, frame_idx]  # [B, N, 6]
            
            # Extract point features
            point_features = self.ptv3(frame_points)  # [B, N, feature_dim]
            all_features.append(point_features)
            
            # Adapt features for CLIP alignment
            adapted_feat = self.adapter(point_features)  # [B, N, clip_dim]
            adapted_features.append(adapted_feat)
        
        # Stack features
        all_features = torch.stack(all_features, dim=1)  # [B, num_frames, N, feature_dim]
        adapted_features = torch.stack(adapted_features, dim=1)  # [B, num_frames, N, clip_dim]
        
        # Generate text descriptions for each frame
        text_features_per_frame = []
        instance_info_per_frame = []
        
        for frame_idx in range(num_frames):
            frame_text_features = []
            frame_instance_info = []
            
            for batch_idx in range(B):
                image = batch['images'][batch_idx, frame_idx]
                mask = batch['instance_masks'][batch_idx, frame_idx]
                
                # Generate descriptions using LLaVA (mock)
                descriptions = self.llava_generator.generate_descriptions(image, mask)
                
                # Encode descriptions with CLIP
                if descriptions:
                    desc_list = list(descriptions.values())
                    text_features = self.clip_encoder.encode_text(desc_list)
                    instance_ids = list(descriptions.keys())
                else:
                    text_features = torch.empty(0, self.clip_encoder.feature_dim).to(image.device)
                    instance_ids = []
                
                frame_text_features.append(text_features)
                frame_instance_info.append({
                    'instance_ids': instance_ids,
                    'descriptions': descriptions
                })
            
            text_features_per_frame.append(frame_text_features)
            instance_info_per_frame.append(frame_instance_info)
        
        return {
            'point_features': all_features,
            'adapted_features': adapted_features,
            'text_features': text_features_per_frame,
            'instance_info': instance_info_per_frame,
            'point_indices': batch['point_indices'],
            'visibility_matrix': batch['visibility_matrix']
        }
    
    def extract_instance_features(self, point_features: torch.Tensor, 
                                instance_mask: torch.Tensor,
                                point_indices: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Extract features for each instance by averaging point features within instance masks
        
        Args:
            point_features: [N, feature_dim] point features
            instance_mask: [H, W] instance mask
            point_indices: [N] mapping from frame points to scene points
        Returns:
            Dictionary mapping instance_id -> feature vector
        """
        H, W = instance_mask.shape
        device = point_features.device
        
        # For simplicity, we'll assume a correspondence between points and pixels
        # In practice, you'd need camera projection to map 3D points to 2D pixels
        
        unique_instances = torch.unique(instance_mask)
        unique_instances = unique_instances[unique_instances > 0]
        
        instance_features = {}
        
        for instance_id in unique_instances:
            instance_id_int = instance_id.item()
            
            # Create mask for this instance
            mask_2d = (instance_mask == instance_id)
            
            # For simplified correspondence, sample random points
            # In practice, use proper 2D-3D correspondence
            mask_points = torch.randperm(len(point_features))[:mask_2d.sum().item()]
            
            if len(mask_points) > 0:
                instance_feat = point_features[mask_points].mean(dim=0)
                instance_features[instance_id_int] = instance_feat
        
        return instance_features
    
    def update_feature_memory(self, scene_id: str, features: Dict[int, torch.Tensor]):
        """Update feature memory for cross-frame consistency"""
        if scene_id not in self.feature_memory:
            self.feature_memory[scene_id] = {}
        
        for instance_id, feat in features.items():
            if instance_id in self.feature_memory[scene_id]:
                # Exponential moving average
                alpha = 0.1
                self.feature_memory[scene_id][instance_id] = (
                    alpha * feat + (1 - alpha) * self.feature_memory[scene_id][instance_id]
                )
            else:
                self.feature_memory[scene_id][instance_id] = feat.clone()
    
    def inference_single_frame(self, points: torch.Tensor, 
                              image: torch.Tensor,
                              scene_id: str) -> Dict:
        """
        Online inference for a single frame
        
        Args:
            points: [N, 6] point cloud
            image: [3, H, W] RGB image  
            scene_id: Scene identifier
        Returns:
            Dictionary with predictions and updated features
        """
        self.eval()
        with torch.no_grad():
            # Add batch dimension
            points = points.unsqueeze(0)  # [1, N, 6]
            
            # Extract point features
            point_features = self.ptv3(points)  # [1, N, feature_dim]
            adapted_features = self.adapter(point_features)  # [1, N, clip_dim]
            
            # Remove batch dimension
            point_features = point_features.squeeze(0)  # [N, feature_dim]
            adapted_features = adapted_features.squeeze(0)  # [N, clip_dim]
            
            return {
                'point_features': point_features,
                'adapted_features': adapted_features,
                'scene_id': scene_id
            }
