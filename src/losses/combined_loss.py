"""
Combined loss function that integrates all three loss components
"""
import torch
import torch.nn as nn
from typing import Dict, List
import logging

from .instance_loss import InstanceDiscriminationLoss
from .semantic_loss import SemanticAlignmentLoss
from .consistency_loss import CrossFrameConsistencyLoss

logger = logging.getLogger(__name__)

class CombinedLoss(nn.Module):
    """
    Combined loss function with three components:
    1. Instance discrimination loss
    2. Semantic alignment loss
    3. Cross-frame consistency loss
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Loss weights from config
        self.weights = config.get('loss_weights', {
            'instance_discrimination': 1.0,
            'semantic_alignment': 1.0,
            'cross_frame_consistency': 0.5
        })
        
        # Initialize individual loss functions
        self.instance_loss = InstanceDiscriminationLoss(
            temperature=0.1,
            margin=1.0,
            intra_weight=1.0,
            inter_weight=1.0,
            compactness_weight=0.5
        )
        
        self.semantic_loss = SemanticAlignmentLoss(
            temperature=0.1,
            margin=0.2
        )
        
        self.consistency_loss = CrossFrameConsistencyLoss(
            temperature=0.1,
            consistency_weight=1.0,
            temporal_weight=0.5
        )
        
        logger.info(f"Initialized CombinedLoss with weights: {self.weights}")
    
    def forward(self, model_output: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss
        
        Args:
            model_output: Dictionary from model forward pass containing:
                - point_features: [B, num_frames, N, feature_dim]
                - adapted_features: [B, num_frames, N, clip_dim]
                - text_features: List of text features per frame
                - instance_info: List of instance information per frame
            batch: Original batch data containing:
                - instance_masks: [B, num_frames, H, W]
                - visibility_matrix: [B, total_points, num_frames]
                - point_indices: List of point index mappings
        Returns:
            Dictionary containing all loss components
        """
        device = model_output['point_features'].device
        B, num_frames, N, feature_dim = model_output['point_features'].shape
        
        # Initialize loss dictionaries
        total_instance_losses = []
        total_semantic_losses = []
        total_consistency_losses = []
        
        # Process each batch
        for batch_idx in range(B):
            # Extract data for this batch
            batch_point_features = model_output['point_features'][batch_idx]  # [num_frames, N, feature_dim]
            batch_adapted_features = model_output['adapted_features'][batch_idx]  # [num_frames, N, clip_dim]
            batch_instance_masks = batch['instance_masks'][batch_idx]  # [num_frames, H, W]
            batch_visibility = batch['visibility_matrix'][batch_idx]  # [total_points, num_frames]
            batch_point_indices = [indices[batch_idx] for indices in model_output['point_indices']]
            
            # Get text features for this batch
            batch_text_features = []
            batch_instance_info = []
            for frame_idx in range(num_frames):
                if (frame_idx < len(model_output['text_features']) and 
                    batch_idx < len(model_output['text_features'][frame_idx])):
                    batch_text_features.append(model_output['text_features'][frame_idx][batch_idx])
                    batch_instance_info.append(model_output['instance_info'][frame_idx][batch_idx])
                else:
                    # Handle missing data
                    batch_text_features.append(torch.empty(0, batch_adapted_features.shape[-1]).to(device))
                    batch_instance_info.append({'instance_ids': [], 'descriptions': {}})
            
            # Compute losses for each frame
            frame_instance_losses = []
            frame_semantic_losses = []
            
            for frame_idx in range(num_frames):
                frame_point_features = batch_point_features[frame_idx]  # [N, feature_dim]
                frame_adapted_features = batch_adapted_features[frame_idx]  # [N, clip_dim]
                frame_mask = batch_instance_masks[frame_idx]  # [H, W]
                frame_point_indices = batch_point_indices[frame_idx]  # [N]
                
                # Instance discrimination loss
                instance_loss_dict = self.instance_loss(
                    frame_point_features,
                    frame_mask,
                    frame_point_indices
                )
                frame_instance_losses.append(instance_loss_dict['total_loss'])
                
                # Semantic alignment loss
                semantic_loss_dict = self.semantic_loss(
                    frame_adapted_features,
                    [batch_text_features[frame_idx]] if frame_idx < len(batch_text_features) else [],
                    [batch_instance_info[frame_idx]] if frame_idx < len(batch_instance_info) else [],
                    frame_mask,
                    frame_point_indices
                )
                frame_semantic_losses.append(semantic_loss_dict['total_loss'])
            
            # Cross-frame consistency loss
            consistency_loss_dict = self.consistency_loss(
                batch_point_features,
                batch_visibility,
                batch_point_indices
            )
            
            # Aggregate losses for this batch
            avg_instance_loss = torch.stack(frame_instance_losses).mean() if frame_instance_losses else torch.tensor(0.0, device=device)
            avg_semantic_loss = torch.stack(frame_semantic_losses).mean() if frame_semantic_losses else torch.tensor(0.0, device=device)
            consistency_loss_val = consistency_loss_dict['total_loss']
            
            total_instance_losses.append(avg_instance_loss)
            total_semantic_losses.append(avg_semantic_loss)
            total_consistency_losses.append(consistency_loss_val)
        
        # Average across batches
        final_instance_loss = torch.stack(total_instance_losses).mean() if total_instance_losses else torch.tensor(0.0, device=device)
        final_semantic_loss = torch.stack(total_semantic_losses).mean() if total_semantic_losses else torch.tensor(0.0, device=device)
        final_consistency_loss = torch.stack(total_consistency_losses).mean() if total_consistency_losses else torch.tensor(0.0, device=device)
        
        # Weighted combination
        total_loss = (
            self.weights['instance_discrimination'] * final_instance_loss +
            self.weights['semantic_alignment'] * final_semantic_loss +
            self.weights['cross_frame_consistency'] * final_consistency_loss
        )
        
        return {
            'total_loss': total_loss,
            'instance_discrimination_loss': final_instance_loss,
            'semantic_alignment_loss': final_semantic_loss,
            'cross_frame_consistency_loss': final_consistency_loss,
            'loss_weights': self.weights
        }
