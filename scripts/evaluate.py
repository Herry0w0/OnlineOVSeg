#!/usr/bin/env python3
"""
Evaluation script for online 3D instance segmentation
"""
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import logging
import json
from pathlib import Path
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

from src.utils import setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate online 3D instance segmentation results')
    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to predictions directory')
    parser.add_argument('--ground-truth', type=str, required=True,
                       help='Path to ground truth directory')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                       help='Output file for evaluation results')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    return parser.parse_args()

def compute_iou(pred_mask, gt_mask):
    """Compute IoU between prediction and ground truth masks"""
    intersection = torch.logical_and(pred_mask, gt_mask).sum()
    union = torch.logical_or(pred_mask, gt_mask).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (intersection / union).item()

def evaluate_instance_segmentation(pred_instances, gt_instances):
    """
    Evaluate instance segmentation performance
    
    Args:
        pred_instances: [N] predicted instance labels
        gt_instances: [N] ground truth instance labels
    Returns:
        Dictionary with evaluation metrics
    """
    # Convert to numpy if torch tensors
    if isinstance(pred_instances, torch.Tensor):
        pred_instances = pred_instances.cpu().numpy()
    if isinstance(gt_instances, torch.Tensor):
        gt_instances = gt_instances.cpu().numpy()
    
    # Remove background (label 0)
    valid_mask = gt_instances > 0
    pred_instances = pred_instances[valid_mask]
    gt_instances = gt_instances[valid_mask]
    
    if len(pred_instances) == 0:
        return {
            'ari': 0.0,
            'ami': 0.0,
            'coverage': 0.0,
            'weighted_coverage': 0.0
        }
    
    # Adjusted Rand Index
    ari = adjusted_rand_score(gt_instances, pred_instances)
    
    # Adjusted Mutual Information
    ami = adjusted_mutual_info_score(gt_instances, pred_instances)
    
    # Coverage metrics
    gt_unique = np.unique(gt_instances)
    pred_unique = np.unique(pred_instances)
    
    # For each GT instance, find best matching prediction
    ious = []
    for gt_id in gt_unique:
        if gt_id == 0:  # Skip background
            continue
            
        gt_mask = (gt_instances == gt_id)
        best_iou = 0.0
        
        for pred_id in pred_unique:
            if pred_id == 0:  # Skip background
                continue
                
            pred_mask = (pred_instances == pred_id)
            iou = compute_iou(torch.tensor(pred_mask), torch.tensor(gt_mask))
            best_iou = max(best_iou, iou)
        
        ious.append(best_iou)
    
    coverage = np.mean(ious) if ious else 0.0
    weighted_coverage = np.mean(ious) if ious else 0.0  # Could weight by instance size
    
    return {
        'ari': ari,
        'ami': ami,
        'coverage': coverage,
        'weighted_coverage': weighted_coverage,
        'num_gt_instances': len(gt_unique) - (1 if 0 in gt_unique else 0),
        'num_pred_instances': len(pred_unique) - (1 if 0 in pred_unique else 0)
    }

def evaluate_semantic_segmentation(pred_labels, gt_labels, label_names=None):
    """
    Evaluate semantic segmentation performance
    """
    # Simple accuracy for now
    if isinstance(pred_labels, list):
        # Convert text labels to indices
        unique_labels = list(set(pred_labels + gt_labels))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        
        pred_indices = [label_to_idx[label] for label in pred_labels]
        gt_indices = [label_to_idx[label] for label in gt_labels]
        
        accuracy = np.mean([p == g for p, g in zip(pred_indices, gt_indices)])
    else:
        accuracy = (pred_labels == gt_labels).float().mean().item()
    
    return {
        'accuracy': accuracy
    }

def evaluate_temporal_consistency(frame_results):
    """
    Evaluate temporal consistency across frames
    """
    if len(frame_results) < 2:
        return {'temporal_consistency': 1.0}
    
    consistency_scores = []
    
    for i in range(len(frame_results) - 1):
        current_instances = frame_results[i]['predictions']['instance_ids']
        next_instances = frame_results[i + 1]['predictions']['instance_ids']
        
        # Simple consistency: how many instances remain the same
        if hasattr(current_instances, 'cpu'):
            current_instances = current_instances.cpu().numpy()
        if hasattr(next_instances, 'cpu'):
            next_instances = next_instances.cpu().numpy()
        
        # Count common instances (simplified)
        current_unique = set(np.unique(current_instances))
        next_unique = set(np.unique(next_instances))
        
        if len(current_unique) == 0 or len(next_unique) == 0:
            consistency = 1.0
        else:
            intersection = len(current_unique.intersection(next_unique))
            union = len(current_unique.union(next_unique))
            consistency = intersection / union if union > 0 else 1.0
        
        consistency_scores.append(consistency)
    
    return {
        'temporal_consistency': np.mean(consistency_scores),
        'min_consistency': np.min(consistency_scores),
        'max_consistency': np.max(consistency_scores)
    }

def load_predictions(pred_path):
    """Load prediction results"""
    pred_path = Path(pred_path)
    
    if pred_path.is_file():
        # Single results file
        if pred_path.suffix == '.json':
            with open(pred_path, 'r') as f:
                return json.load(f)
        elif pred_path.suffix == '.pth':
            return torch.load(pred_path)
    
    elif pred_path.is_dir():
        # Directory with multiple scenes
        results = {}
        for scene_dir in pred_path.iterdir():
            if scene_dir.is_dir():
                scene_results = []
                pred_files = sorted(list(scene_dir.glob("*_predictions.pth")))
                
                for pred_file in pred_files:
                    frame_result = torch.load(pred_file)
                    scene_results.append(frame_result)
                
                if scene_results:
                    results[scene_dir.name] = scene_results
        
        return results
    
    else:
        raise ValueError(f"Invalid prediction path: {pred_path}")

def load_ground_truth(gt_path, scene_id=None):
    """Load ground truth data (simplified for demo)"""
    # For demo purposes, create dummy ground truth
    # In practice, this would load actual ScanNet annotations
    
    dummy_gt = {
        'instance_ids': torch.randint(0, 10, (1000,)),
        'semantic_labels': ['chair'] * 200 + ['table'] * 300 + ['wall'] * 500
    }
    
    return dummy_gt

def main():
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("Starting evaluation")
    
    try:
        # Load predictions
        logger.info(f"Loading predictions from {args.predictions}")
        predictions = load_predictions(args.predictions)
        
        # Initialize evaluation results
        eval_results = {
            'overall': {
                'instance_metrics': [],
                'semantic_metrics': [],
                'temporal_metrics': []
            },
            'per_scene': {}
        }
        
        # Evaluate each scene
        if isinstance(predictions, dict):
            # Multiple scenes
            for scene_id, scene_results in predictions.items():
                logger.info(f"Evaluating scene: {scene_id}")
                
                # Load ground truth for this scene
                gt_data = load_ground_truth(args.ground_truth, scene_id)
                
                # Evaluate instance segmentation
                instance_metrics = []
                semantic_metrics = []
                
                for frame_result in scene_results:
                    if 'predictions' in frame_result:
                        pred = frame_result['predictions']
                        
                        # Instance evaluation
                        inst_metrics = evaluate_instance_segmentation(
                            pred['instance_ids'], 
                            gt_data['instance_ids']
                        )
                        instance_metrics.append(inst_metrics)
                        
                        # Semantic evaluation
                        sem_metrics = evaluate_semantic_segmentation(
                            pred.get('semantic_labels', []), 
                            gt_data['semantic_labels']
                        )
                        semantic_metrics.append(sem_metrics)
                
                # Temporal consistency
                temporal_metrics = evaluate_temporal_consistency(scene_results)
                
                # Aggregate scene metrics
                scene_eval = {
                    'instance_metrics': {
                        'mean_ari': np.mean([m['ari'] for m in instance_metrics]),
                        'mean_ami': np.mean([m['ami'] for m in instance_metrics]),
                        'mean_coverage': np.mean([m['coverage'] for m in instance_metrics])
                    },
                    'semantic_metrics': {
                        'mean_accuracy': np.mean([m['accuracy'] for m in semantic_metrics])
                    },
                    'temporal_metrics': temporal_metrics,
                    'num_frames': len(scene_results)
                }
                
                eval_results['per_scene'][scene_id] = scene_eval
                eval_results['overall']['instance_metrics'].extend(instance_metrics)
                eval_results['overall']['semantic_metrics'].extend(semantic_metrics)
                eval_results['overall']['temporal_metrics'].append(temporal_metrics)
        
        else:
            # Single scene/sequence
            logger.info("Evaluating single sequence")
            # Similar evaluation logic for single sequence
            pass
        
        # Compute overall metrics
        if eval_results['overall']['instance_metrics']:
            overall_instance = {
                'mean_ari': np.mean([m['ari'] for m in eval_results['overall']['instance_metrics']]),
                'mean_ami': np.mean([m['ami'] for m in eval_results['overall']['instance_metrics']]),
                'mean_coverage': np.mean([m['coverage'] for m in eval_results['overall']['instance_metrics']])
            }
            eval_results['overall']['instance_summary'] = overall_instance
        
        if eval_results['overall']['semantic_metrics']:
            overall_semantic = {
                'mean_accuracy': np.mean([m['accuracy'] for m in eval_results['overall']['semantic_metrics']])
            }
            eval_results['overall']['semantic_summary'] = overall_semantic
        
        if eval_results['overall']['temporal_metrics']:
            overall_temporal = {
                'mean_consistency': np.mean([m['temporal_consistency'] for m in eval_results['overall']['temporal_metrics']])
            }
            eval_results['overall']['temporal_summary'] = overall_temporal
        
        # Save results
        with open(args.output, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.float64):
                    return float(obj)
                elif isinstance(obj, np.int64):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(v) for v in obj]
                return obj
            
            json.dump(convert_numpy(eval_results), f, indent=2)
        
        logger.info(f"Evaluation completed! Results saved to {args.output}")
        
        # Print summary
        if 'instance_summary' in eval_results['overall']:
            inst_summary = eval_results['overall']['instance_summary']
            logger.info(f"Instance Segmentation - ARI: {inst_summary['mean_ari']:.4f}, "
                       f"AMI: {inst_summary['mean_ami']:.4f}, "
                       f"Coverage: {inst_summary['mean_coverage']:.4f}")
        
        if 'semantic_summary' in eval_results['overall']:
            sem_summary = eval_results['overall']['semantic_summary']
            logger.info(f"Semantic Segmentation - Accuracy: {sem_summary['mean_accuracy']:.4f}")
        
        if 'temporal_summary' in eval_results['overall']:
            temp_summary = eval_results['overall']['temporal_summary']
            logger.info(f"Temporal Consistency: {temp_summary['mean_consistency']:.4f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())
