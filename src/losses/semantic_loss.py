"""
Semantic alignment loss for aligning point features with CLIP text features
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class SemanticAlignmentLoss(nn.Module):
    """
    Semantic alignment loss between point features and CLIP text features
    """
    
    def __init__(self, 
                 temperature: float = 0.1,
                 margin: float = 0.2):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
    
    def forward(self, 
                adapted_features: torch.Tensor,
                text_features: List[torch.Tensor],
                instance_info: List[Dict],
                instance_masks: torch.Tensor,
                point_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute semantic alignment loss
        
        Args:
            adapted_features: [N, clip_dim] adapted point features
            text_features: List of [num_instances, clip_dim] text features per batch
            instance_info: List of dicts with instance information per batch
            instance_masks: [H, W] instance masks
            point_indices: [N] point indices
        Returns:
            Dictionary containing loss components
        """
        device = adapted_features.device
        
        if not text_features or len(text_features) == 0:
            return {
                'alignment_loss': torch.tensor(0.0, device=device),
                'contrastive_loss': torch.tensor(0.0, device=device),
                'total_loss': torch.tensor(0.0, device=device)
            }
        
        # For single batch processing
        batch_text_features = text_features[0] if text_features else torch.empty(0, adapted_features.shape[-1]).to(device)
        batch_instance_info = instance_info[0] if instance_info else {'instance_ids': []}
        
        if len(batch_text_features) == 0:
            return {
                'alignment_loss': torch.tensor(0.0, device=device),
                'contrastive_loss': torch.tensor(0.0, device=device),
                'total_loss': torch.tensor(0.0, device=device)
            }
        
        H, W = instance_masks.shape
        N, clip_dim = adapted_features.shape
        
        alignment_losses = []
        contrastive_losses = []
        
        # Process each instance
        instance_ids = batch_instance_info['instance_ids']
        
        for i, instance_id in enumerate(instance_ids):
            if i >= len(batch_text_features):
                break
                
            # Get text feature for this instance
            text_feat = batch_text_features[i]  # [clip_dim]
            
            # Get point features for this instance
            instance_mask = (instance_masks == instance_id)
            
            if not instance_mask.any():
                continue
            
            # Sample points for this instance (simplified correspondence)
            mask_pixels = torch.nonzero(instance_mask, as_tuple=False)
            num_points = min(len(mask_pixels), N // len(instance_ids))
            
            if num_points > 0:
                sampled_indices = torch.randperm(N, device=device)[:num_points]
                instance_point_features = adapted_features[sampled_indices]  # [num_points, clip_dim]
                
                # Compute alignment loss (cosine similarity)
                similarities = self.cosine_similarity(
                    instance_point_features, 
                    text_feat.unsqueeze(0).expand_as(instance_point_features)
                )
                
                # Encourage high similarity
                alignment_loss = (1 - similarities).mean()
                alignment_losses.append(alignment_loss)
                
                # Contrastive loss against other text features
                if len(batch_text_features) > 1:
                    # Positive similarity (current instance)
                    pos_sim = similarities.mean()
                    
                    # Negative similarities (other instances)
                    other_text_features = torch.cat([
                        batch_text_features[:i], 
                        batch_text_features[i+1:]
                    ], dim=0) if i < len(batch_text_features) - 1 else batch_text_features[:i]
                    
                    if len(other_text_features) > 0:
                        neg_sims = []
                        for other_text_feat in other_text_features:
                            neg_sim = self.cosine_similarity(
                                instance_point_features.mean(dim=0, keepdim=True),
                                other_text_feat.unsqueeze(0)
                            )
                            neg_sims.append(neg_sim)
                        
                        neg_sims = torch.stack(neg_sims)
                        
                        # Contrastive loss
                        contrastive_loss = F.relu(neg_sims - pos_sim + self.margin).mean()
                        contrastive_losses.append(contrastive_loss)
        
        # Aggregate losses
        if alignment_losses:
            avg_alignment_loss = torch.stack(alignment_losses).mean()
        else:
            avg_alignment_loss = torch.tensor(0.0, device=device)
        
        if contrastive_losses:
            avg_contrastive_loss = torch.stack(contrastive_losses).mean()
        else:
            avg_contrastive_loss = torch.tensor(0.0, device=device)
        
        total_loss = avg_alignment_loss + avg_contrastive_loss
        
        return {
            'alignment_loss': avg_alignment_loss,
            'contrastive_loss': avg_contrastive_loss,
            'total_loss': total_loss
        }

class CLIPSimilarityLoss(nn.Module):
    """CLIP-style similarity loss"""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, point_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Compute CLIP-style contrastive loss
        
        Args:
            point_features: [N, dim] normalized point features
            text_features: [M, dim] normalized text features
        Returns:
            Contrastive loss
        """
        device = point_features.device
        
        # Compute similarity matrix
        logits = torch.matmul(point_features, text_features.T) / self.temperature
        
        # Create targets (assuming one-to-one correspondence)
        N, M = logits.shape
        if N != M:
            # Handle dimension mismatch
            min_dim = min(N, M)
            logits = logits[:min_dim, :min_dim]
            N = M = min_dim
        
        targets = torch.arange(N, device=device)
        
        # Symmetric loss
        loss_i = F.cross_entropy(logits, targets)
        loss_t = F.cross_entropy(logits.T, targets)
        
        return (loss_i + loss_t) / 2
