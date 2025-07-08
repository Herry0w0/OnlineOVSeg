"""
Cross-frame consistency loss for maintaining temporal coherence
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class CrossFrameConsistencyLoss(nn.Module):
    """
    Cross-frame consistency loss for temporal coherence
    """
    
    def __init__(self, 
                 temperature: float = 0.1,
                 consistency_weight: float = 1.0,
                 temporal_weight: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.consistency_weight = consistency_weight
        self.temporal_weight = temporal_weight
    
    def forward(self, 
                point_features: torch.Tensor,
                visibility_matrix: torch.Tensor,
                point_indices: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute cross-frame consistency loss
        
        Args:
            point_features: [num_frames, N, feature_dim] point features across frames
            visibility_matrix: [total_points, num_frames] visibility of scene points
            point_indices: List of [N] tensors mapping frame points to scene points
        Returns:
            Dictionary containing loss components
        """
        device = point_features.device
        num_frames, N, feature_dim = point_features.shape
        total_scene_points = visibility_matrix.shape[0]
        
        if num_frames < 2:
            return {
                'consistency_loss': torch.tensor(0.0, device=device),
                'temporal_smoothness_loss': torch.tensor(0.0, device=device),
                'total_loss': torch.tensor(0.0, device=device)
            }
        
        # Find overlapping points between frames
        consistency_losses = []
        temporal_losses = []
        
        # Consistency between adjacent frames
        for i in range(num_frames - 1):
            current_indices = point_indices[i]  # [N]
            next_indices = point_indices[i + 1]  # [N]
            
            current_features = point_features[i]  # [N, feature_dim]
            next_features = point_features[i + 1]  # [N, feature_dim]
            
            # Find overlapping scene points
            overlap_loss = self._compute_overlap_consistency(
                current_features, next_features,
                current_indices, next_indices,
                visibility_matrix
            )
            
            if overlap_loss is not None:
                consistency_losses.append(overlap_loss)
        
        # Temporal smoothness across all frames
        temporal_loss = self._compute_temporal_smoothness(
            point_features, visibility_matrix, point_indices
        )
        temporal_losses.append(temporal_loss)
        
        # Aggregate losses
        consistency_loss = torch.stack(consistency_losses).mean() if consistency_losses else torch.tensor(0.0, device=device)
        temporal_smoothness_loss = torch.stack(temporal_losses).mean() if temporal_losses else torch.tensor(0.0, device=device)
        
        total_loss = (self.consistency_weight * consistency_loss + 
                     self.temporal_weight * temporal_smoothness_loss)
        
        return {
            'consistency_loss': consistency_loss,
            'temporal_smoothness_loss': temporal_smoothness_loss,
            'total_loss': total_loss
        }
    
    def _compute_overlap_consistency(self,
                                   features1: torch.Tensor,
                                   features2: torch.Tensor,
                                   indices1: torch.Tensor,
                                   indices2: torch.Tensor,
                                   visibility_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute consistency loss for overlapping points between two frames
        """
        device = features1.device
        
        # Remove padding indices (-1)
        valid_mask1 = indices1 >= 0
        valid_mask2 = indices2 >= 0
        
        valid_indices1 = indices1[valid_mask1]
        valid_indices2 = indices2[valid_mask2]
        valid_features1 = features1[valid_mask1]
        valid_features2 = features2[valid_mask2]
        
        if len(valid_indices1) == 0 or len(valid_indices2) == 0:
            return torch.tensor(0.0, device=device)
        
        # Find common scene points
        common_indices = torch.intersect1d(valid_indices1, valid_indices2)
        
        if len(common_indices) == 0:
            return torch.tensor(0.0, device=device)
        
        # Get features for common points
        mask1 = torch.isin(valid_indices1, common_indices)
        mask2 = torch.isin(valid_indices2, common_indices)
        
        common_features1 = valid_features1[mask1]
        common_features2 = valid_features2[mask2]
        
        # Ensure same ordering
        _, indices_in_common1 = torch.sort(torch.searchsorted(common_indices, valid_indices1[mask1]))
        _, indices_in_common2 = torch.sort(torch.searchsorted(common_indices, valid_indices2[mask2]))
        
        common_features1 = common_features1[indices_in_common1]
        common_features2 = common_features2[indices_in_common2]
        
        # Compute consistency loss
        consistency_loss = F.mse_loss(common_features1, common_features2)
        
        return consistency_loss
    
    def _compute_temporal_smoothness(self,
                                   point_features: torch.Tensor,
                                   visibility_matrix: torch.Tensor,
                                   point_indices: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute temporal smoothness loss across all frames
        """
        device = point_features.device
        num_frames, N, feature_dim = point_features.shape
        
        # Aggregate features for each scene point across all frames
        total_scene_points = visibility_matrix.shape[0]
        scene_features = torch.zeros(total_scene_points, feature_dim, device=device)
        scene_counts = torch.zeros(total_scene_points, device=device)
        
        # Accumulate features for each scene point
        for frame_idx in range(num_frames):
            frame_indices = point_indices[frame_idx]
            frame_features = point_features[frame_idx]
            
            valid_mask = frame_indices >= 0
            valid_indices = frame_indices[valid_mask]
            valid_features = frame_features[valid_mask]
            
            for i, scene_idx in enumerate(valid_indices):
                scene_features[scene_idx] += valid_features[i]
                scene_counts[scene_idx] += 1
        
        # Average features for each scene point
        nonzero_mask = scene_counts > 0
        scene_features[nonzero_mask] /= scene_counts[nonzero_mask].unsqueeze(1)
        
        # Compute temporal smoothness
        smoothness_losses = []
        
        for frame_idx in range(num_frames):
            frame_indices = point_indices[frame_idx]
            frame_features = point_features[frame_idx]
            
            valid_mask = frame_indices >= 0
            valid_indices = frame_indices[valid_mask]
            valid_features = frame_features[valid_mask]
            
            if len(valid_indices) > 0:
                corresponding_scene_features = scene_features[valid_indices]
                smoothness_loss = F.mse_loss(valid_features, corresponding_scene_features)
                smoothness_losses.append(smoothness_loss)
        
        if smoothness_losses:
            return torch.stack(smoothness_losses).mean()
        else:
            return torch.tensor(0.0, device=device)

class TemporalContrastiveLoss(nn.Module):
    """Temporal contrastive loss for cross-frame consistency"""
    
    def __init__(self, temperature: float = 0.1, max_temporal_distance: int = 3):
        super().__init__()
        self.temperature = temperature
        self.max_temporal_distance = max_temporal_distance
    
    def forward(self, features: torch.Tensor, temporal_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [N, feature_dim] features from all frames
            temporal_ids: [N] temporal frame ids for each feature
        Returns:
            Temporal contrastive loss
        """
        device = features.device
        N = features.shape[0]
        
        if N <= 1:
            return torch.tensor(0.0, device=device)
        
        # Compute temporal distances
        temporal_dists = torch.abs(temporal_ids.unsqueeze(1) - temporal_ids.unsqueeze(0))
        
        # Create positive mask (same temporal context)
        pos_mask = (temporal_dists <= self.max_temporal_distance) & (temporal_dists > 0)
        
        # Compute similarities
        similarities = torch.matmul(F.normalize(features), F.normalize(features).T) / self.temperature
        
        # Contrastive loss
        exp_similarities = torch.exp(similarities)
        log_prob = similarities - torch.log(exp_similarities.sum(dim=1, keepdim=True))
        
        mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)
        loss = -mean_log_prob_pos.mean()
        
        return loss
