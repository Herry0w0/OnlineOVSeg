"""
Online inference engine for real-time 3D instance segmentation
"""
import torch
import torch.nn.functional as F
import numpy as np
import os
import cv2
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict
import time

from ..models import OnlineInstanceSegmentationModel
from ..utils import load_checkpoint

logger = logging.getLogger(__name__)

class OnlineInferenceEngine:
    """
    Online inference engine for real-time 3D instance segmentation
    Maintains temporal consistency and updates features incrementally
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config['hardware']['device'])
        
        # Load model
        self.model = OnlineInstanceSegmentationModel(config['model']).to(self.device)
        self._load_model_checkpoint()
        self.model.eval()
        
        # Initialize online state
        self.scene_features = {}  # scene_id -> Dict[point_idx, feature]
        self.instance_memory = {}  # scene_id -> Dict[instance_id, feature]
        self.frame_history = {}  # scene_id -> List[frame_data]
        self.temporal_features = {}  # scene_id -> temporal feature buffer
        
        # Inference parameters
        self.consistency_threshold = config['inference']['consistency_threshold']
        self.feature_update_rate = config['inference']['feature_update_rate']
        self.max_history_length = 10
        
        # Output configuration
        self.save_results = config['output']['save_results']
        self.output_dir = config['output']['output_dir']
        if self.save_results:
            os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info("OnlineInferenceEngine initialized")
    
    def _load_model_checkpoint(self):
        """Load trained model checkpoint"""
        checkpoint_path = self.config['model']['checkpoint_path']
        if os.path.exists(checkpoint_path):
            epoch, loss = load_checkpoint(checkpoint_path, self.model)
            logger.info(f"Loaded model checkpoint from epoch {epoch} with loss {loss:.4f}")
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
    
    def process_frame(self, 
                     points: torch.Tensor,
                     image: torch.Tensor,
                     scene_id: str,
                     frame_id: int) -> Dict:
        """
        Process a single frame in online mode
        
        Args:
            points: [N, 6] point cloud (xyz + rgb)
            image: [3, H, W] RGB image
            scene_id: Scene identifier
            frame_id: Frame identifier
        Returns:
            Dictionary with predictions and features
        """
        start_time = time.time()
        
        with torch.no_grad():
            # Move to device
            points = points.to(self.device)
            image = image.to(self.device)
            
            # Extract features
            frame_output = self.model.inference_single_frame(points, image, scene_id)
            
            # Update online state
            self._update_online_state(frame_output, scene_id, frame_id)
            
            # Predict instances
            predictions = self._predict_instances(frame_output, scene_id, frame_id)
            
            # Update feature memory
            self._update_feature_memory(predictions, scene_id, frame_id)
            
            # Temporal consistency enforcement
            if self.config['inference']['online_update']:
                predictions = self._enforce_temporal_consistency(predictions, scene_id, frame_id)
        
        processing_time = time.time() - start_time
        
        # Prepare output
        result = {
            'predictions': predictions,
            'features': frame_output,
            'scene_id': scene_id,
            'frame_id': frame_id,
            'processing_time': processing_time
        }
        
        # Save results if requested
        if self.save_results:
            self._save_frame_results(result)
        
        return result
    
    def _update_online_state(self, frame_output: Dict, scene_id: str, frame_id: int):
        """Update online state with new frame"""
        if scene_id not in self.scene_features:
            self.scene_features[scene_id] = {}
            self.frame_history[scene_id] = []
            self.temporal_features[scene_id] = []
        
        # Store frame data
        frame_data = {
            'frame_id': frame_id,
            'features': frame_output['point_features'].cpu(),
            'adapted_features': frame_output['adapted_features'].cpu(),
            'timestamp': time.time()
        }
        
        self.frame_history[scene_id].append(frame_data)
        
        # Maintain history limit
        if len(self.frame_history[scene_id]) > self.max_history_length:
            self.frame_history[scene_id].pop(0)
        
        # Update temporal features buffer
        self.temporal_features[scene_id].append(frame_output['adapted_features'])
        if len(self.temporal_features[scene_id]) > self.max_history_length:
            self.temporal_features[scene_id].pop(0)
    
    def _predict_instances(self, frame_output: Dict, scene_id: str, frame_id: int) -> Dict:
        """
        Predict instances for current frame
        """
        point_features = frame_output['point_features']  # [N, feature_dim]
        adapted_features = frame_output['adapted_features']  # [N, clip_dim]
        
        N, feature_dim = point_features.shape
        
        # Simple clustering-based instance prediction
        # In practice, you might use more sophisticated methods
        predictions = self._cluster_instances(adapted_features)
        
        # Assign semantic labels (simplified)
        semantic_labels = self._assign_semantic_labels(adapted_features, predictions['instance_ids'])
        
        return {
            'instance_ids': predictions['instance_ids'],
            'instance_scores': predictions['instance_scores'],
            'semantic_labels': semantic_labels,
            'point_features': point_features,
            'adapted_features': adapted_features
        }
    
    def _cluster_instances(self, features: torch.Tensor) -> Dict:
        """
        Simple feature clustering for instance prediction
        """
        N, dim = features.shape
        device = features.device
        
        # Normalize features
        features_norm = F.normalize(features, p=2, dim=1)
        
        # Simple k-means style clustering (simplified)
        # In practice, use more sophisticated clustering
        num_clusters = min(10, N // 50 + 1)  # Adaptive number of clusters
        
        # Random initialization of centroids
        centroids = features_norm[torch.randperm(N)[:num_clusters]]
        
        # Simple assignment
        similarities = torch.mm(features_norm, centroids.T)
        instance_ids = torch.argmax(similarities, dim=1)
        instance_scores = torch.max(similarities, dim=1)[0]
        
        return {
            'instance_ids': instance_ids,
            'instance_scores': instance_scores,
            'centroids': centroids
        }
    
    def _assign_semantic_labels(self, features: torch.Tensor, instance_ids: torch.Tensor) -> List[str]:
        """
        Assign semantic labels to instances (simplified)
        """
        unique_instances = torch.unique(instance_ids)
        semantic_labels = []
        
        # Simple mapping based on feature characteristics
        label_map = {
            0: "chair", 1: "table", 2: "monitor", 3: "book", 4: "plant",
            5: "wall", 6: "floor", 7: "lamp", 8: "door", 9: "ceiling"
        }
        
        for point_idx in range(len(instance_ids)):
            instance_id = instance_ids[point_idx].item()
            label = label_map.get(instance_id % len(label_map), "object")
            semantic_labels.append(label)
        
        return semantic_labels
    
    def _update_feature_memory(self, predictions: Dict, scene_id: str, frame_id: int):
        """Update feature memory with new predictions"""
        if scene_id not in self.instance_memory:
            self.instance_memory[scene_id] = {}
        
        instance_ids = predictions['instance_ids']
        features = predictions['adapted_features']
        
        # Compute instance centroids
        unique_instances = torch.unique(instance_ids)
        
        for instance_id in unique_instances:
            instance_id_int = instance_id.item()
            mask = (instance_ids == instance_id)
            instance_features = features[mask]
            
            if len(instance_features) > 0:
                centroid = instance_features.mean(dim=0)
                
                if instance_id_int in self.instance_memory[scene_id]:
                    # Exponential moving average
                    alpha = self.feature_update_rate
                    self.instance_memory[scene_id][instance_id_int] = (
                        alpha * centroid + 
                        (1 - alpha) * self.instance_memory[scene_id][instance_id_int]
                    )
                else:
                    self.instance_memory[scene_id][instance_id_int] = centroid
    
    def _enforce_temporal_consistency(self, predictions: Dict, scene_id: str, frame_id: int) -> Dict:
        """
        Enforce temporal consistency with previous frames
        """
        if (scene_id not in self.temporal_features or 
            len(self.temporal_features[scene_id]) < 2):
            return predictions
        
        current_features = predictions['adapted_features']
        current_ids = predictions['instance_ids']
        
        # Get previous frame features
        prev_features = self.temporal_features[scene_id][-2]
        
        # Compute consistency scores
        consistency_scores = self._compute_consistency_scores(
            current_features, prev_features, current_ids
        )
        
        # Update predictions based on consistency
        updated_predictions = self._update_predictions_with_consistency(
            predictions, consistency_scores
        )
        
        return updated_predictions
    
    def _compute_consistency_scores(self, 
                                  current_features: torch.Tensor,
                                  prev_features: torch.Tensor,
                                  current_ids: torch.Tensor) -> torch.Tensor:
        """Compute temporal consistency scores"""
        # Simple consistency score based on feature similarity
        similarities = torch.mm(
            F.normalize(current_features, p=2, dim=1),
            F.normalize(prev_features, p=2, dim=1).T
        )
        
        # Max similarity for each current point
        consistency_scores, _ = torch.max(similarities, dim=1)
        
        return consistency_scores
    
    def _update_predictions_with_consistency(self, 
                                           predictions: Dict,
                                           consistency_scores: torch.Tensor) -> Dict:
        """Update predictions based on consistency scores"""
        # Filter out inconsistent predictions
        consistent_mask = consistency_scores > self.consistency_threshold
        
        updated_predictions = {
            'instance_ids': predictions['instance_ids'],
            'instance_scores': predictions['instance_scores'] * consistency_scores,
            'semantic_labels': predictions['semantic_labels'],
            'point_features': predictions['point_features'],
            'adapted_features': predictions['adapted_features'],
            'consistency_scores': consistency_scores,
            'consistent_mask': consistent_mask
        }
        
        return updated_predictions
    
    def _save_frame_results(self, result: Dict):
        """Save frame results to disk"""
        scene_id = result['scene_id']
        frame_id = result['frame_id']
        
        # Create scene directory
        scene_dir = os.path.join(self.output_dir, scene_id)
        os.makedirs(scene_dir, exist_ok=True)
        
        # Save predictions
        predictions_path = os.path.join(scene_dir, f"frame_{frame_id:06d}_predictions.pth")
        torch.save(result['predictions'], predictions_path)
        
        # Save features (optional)
        if self.config['output'].get('save_features', False):
            features_path = os.path.join(scene_dir, f"frame_{frame_id:06d}_features.pth")
            torch.save(result['features'], features_path)
    
    def process_sequence(self, sequence_data: List[Dict]) -> List[Dict]:
        """
        Process a sequence of frames
        
        Args:
            sequence_data: List of dicts with 'points', 'image', 'scene_id', 'frame_id'
        Returns:
            List of prediction results
        """
        results = []
        
        logger.info(f"Processing sequence of {len(sequence_data)} frames")
        
        for i, frame_data in enumerate(sequence_data):
            logger.info(f"Processing frame {i+1}/{len(sequence_data)}")
            
            result = self.process_frame(
                frame_data['points'],
                frame_data['image'],
                frame_data['scene_id'],
                frame_data['frame_id']
            )
            
            results.append(result)
        
        return results
    
    def get_scene_summary(self, scene_id: str) -> Dict:
        """Get summary of processed scene"""
        if scene_id not in self.scene_features:
            return {}
        
        return {
            'num_frames_processed': len(self.frame_history[scene_id]),
            'num_instances': len(self.instance_memory[scene_id]),
            'instance_ids': list(self.instance_memory[scene_id].keys()),
            'last_frame_timestamp': self.frame_history[scene_id][-1]['timestamp'] if self.frame_history[scene_id] else None
        }
    
    def reset_scene(self, scene_id: str):
        """Reset state for a specific scene"""
        if scene_id in self.scene_features:
            del self.scene_features[scene_id]
        if scene_id in self.instance_memory:
            del self.instance_memory[scene_id]
        if scene_id in self.frame_history:
            del self.frame_history[scene_id]
        if scene_id in self.temporal_features:
            del self.temporal_features[scene_id]
        
        logger.info(f"Reset state for scene: {scene_id}")
