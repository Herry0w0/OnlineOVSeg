"""
Instance discrimination loss for encouraging same-instance features to be similar
and different-instance features to be dissimilar
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class InstanceDiscriminationLoss(nn.Module):
    """
    Instance discrimination loss with contrastive learning
    """
    
    def __init__(self, 
                 temperature: float = 0.1,
                 margin: float = 1.0,
                 intra_weight: float = 1.0,
                 inter_weight: float = 1.0,
                 compactness_weight: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.intra_weight = intra_weight
        self.inter_weight = inter_weight
        self.compactness_weight = compactness_weight
    
    def forward(self, point_features: torch.Tensor,
                instance_masks: torch.Tensor,
                point_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute instance discrimination loss
        
        Args:
            point_features: [N, feature_dim] point features
            instance_masks: [H, W] instance segmentation mask
            point_indices: [N] mapping from frame points to scene points
        Returns:
            Dictionary containing loss components
        """
        device = point_features.device
        H, W = instance_masks.shape
        N, feature_dim = point_features.shape
        
        # Get unique instances (excluding background)
        unique_instances = torch.unique(instance_masks)
        unique_instances = unique_instances[unique_instances > 0]
        
        if len(unique_instances) < 2:
            # Need at least 2 instances for discrimination
            return {
                'intra_instance_loss': torch.tensor(0.0, device=device),
                'inter_instance_loss': torch.tensor(0.0, device=device),
                'compactness_loss': torch.tensor(0.0, device=device),
                'total_loss': torch.tensor(0.0, device=device)
            }
        
        # Extract instance features and compute centroids
        instance_features = {}
        instance_centroids = {}
        
        for instance_id in unique_instances:
            # Get mask for this instance
            instance_mask = (instance_masks == instance_id)
            mask_pixels = torch.nonzero(instance_mask, as_tuple=False)
            
            # For simplicity, randomly sample points for this instance
            # In practice, use proper 2D-3D correspondence
            num_mask_pixels = len(mask_pixels)
            if num_mask_pixels > 0:
                # Sample random subset of points to represent this instance
                num_points = min(num_mask_pixels, N // len(unique_instances))
                sampled_indices = torch.randperm(N, device=device)[:num_points]
                
                instance_feat = point_features[sampled_indices]
                instance_features[instance_id.item()] = instance_feat
                instance_centroids[instance_id.item()] = instance_feat.mean(dim=0)
        
        # Compute intra-instance loss (compactness)
        intra_loss = 0.0
        compactness_loss = 0.0
        num_instances = len(instance_features)
        
        for instance_id, features in instance_features.items():
            centroid = instance_centroids[instance_id]
            
            # Intra-instance similarity (features should be close to centroid)
            distances = F.mse_loss(features, centroid.unsqueeze(0).expand_as(features))
            intra_loss += distances
            
            # Compactness loss (all points should be close to centroid)
            compactness_loss += distances
        
        intra_loss = intra_loss / num_instances if num_instances > 0 else 0.0
        compactness_loss = compactness_loss / num_instances if num_instances > 0 else 0.0
        
        # Compute inter-instance loss (separation)
        inter_loss = 0.0
        centroid_list = list(instance_centroids.values())
        
        if len(centroid_list) > 1:
            centroids = torch.stack(centroid_list)  # [num_instances, feature_dim]
            
            # Compute pairwise similarities
            similarities = F.cosine_similarity(
                centroids.unsqueeze(1), 
                centroids.unsqueeze(0), 
                dim=2
            )
            
            # Remove diagonal (self-similarity)
            mask = ~torch.eye(len(centroids), dtype=torch.bool, device=device)
            similarities = similarities[mask]
            
            # Encourage low similarity between different instances
            inter_loss = F.relu(similarities - (-self.margin)).mean()
        
        # Combine losses
        total_loss = (self.intra_weight * intra_loss + 
                     self.inter_weight * inter_loss +
                     self.compactness_weight * compactness_loss)
        
        return {
            'intra_instance_loss': torch.tensor(intra_loss, device=device),
            'inter_instance_loss': torch.tensor(inter_loss, device=device), 
            'compactness_loss': torch.tensor(compactness_loss, device=device),
            'total_loss': total_loss
        }

class ContrastiveLoss(nn.Module):
    """Contrastive loss for instance discrimination"""
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [N, feature_dim] normalized features
            labels: [N] instance labels
        Returns:
            Contrastive loss
        """
        device = features.device
        N = features.shape[0]
        
        if N <= 1:
            return torch.tensor(0.0, device=device)
        
        # Compute similarity matrix
        similarities = torch.matmul(features, features.T) / self.temperature
        
        # Create positive mask (same instance)
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # Remove diagonal
        mask = mask - torch.eye(N, device=device)
        
        # Compute InfoNCE loss
        exp_similarities = torch.exp(similarities)
        sum_exp_similarities = exp_similarities.sum(dim=1, keepdim=True)
        
        log_prob = similarities - torch.log(sum_exp_similarities)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        loss = -mean_log_prob_pos.mean()
        
        return loss
