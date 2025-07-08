"""
Losses module initialization
"""
from .instance_loss import InstanceDiscriminationLoss, ContrastiveLoss
from .semantic_loss import SemanticAlignmentLoss, CLIPSimilarityLoss
from .consistency_loss import CrossFrameConsistencyLoss, TemporalContrastiveLoss
from .combined_loss import CombinedLoss

__all__ = [
    'InstanceDiscriminationLoss',
    'ContrastiveLoss',
    'SemanticAlignmentLoss', 
    'CLIPSimilarityLoss',
    'CrossFrameConsistencyLoss',
    'TemporalContrastiveLoss',
    'CombinedLoss'
]
