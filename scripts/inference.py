#!/usr/bin/env python3
"""
Inference script for online 3D instance segmentation
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

from src.utils import load_config, setup_logging
from src.inference import OnlineInferenceEngine

def parse_args():
    parser = argparse.ArgumentParser(description='Run online 3D instance segmentation inference')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to inference configuration file')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input data (scene directory or sequence file)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory (overrides config)')
    parser.add_argument('--scene-id', type=str, default=None,
                       help='Scene ID for processing (overrides config)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--visualize', action='store_true',
                       help='Enable visualization')
    return parser.parse_args()

def load_frame_data(input_path: str, scene_id: str) -> list:
    """
    Load frame data from input path
    """
    input_path = Path(input_path)
    
    if input_path.is_file() and input_path.suffix == '.json':
        # Load from sequence file
        with open(input_path, 'r') as f:
            sequence_data = json.load(f)
        return sequence_data
    
    elif input_path.is_dir():
        # Load from scene directory
        scene_dir = input_path / scene_id if scene_id else input_path
        
        if not scene_dir.exists():
            raise ValueError(f"Scene directory not found: {scene_dir}")
        
        # Find point cloud and image files
        color_dir = scene_dir / "color"
        depth_dir = scene_dir / "depth" 
        
        if not color_dir.exists():
            raise ValueError(f"Color directory not found: {color_dir}")
        
        # Get frame list
        image_files = sorted(list(color_dir.glob("*.jpg")))
        
        sequence_data = []
        for i, img_file in enumerate(image_files):
            frame_id = int(img_file.stem)
            
            # Create dummy point cloud data (in practice, load from depth/pose)
            points = torch.randn(1000, 6)  # xyz + rgb
            
            # Load image
            import cv2
            from PIL import Image
            image = Image.open(img_file).convert('RGB')
            image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0
            
            sequence_data.append({
                'points': points,
                'image': image,
                'scene_id': scene_id or scene_dir.name,
                'frame_id': frame_id
            })
        
        return sequence_data
    
    else:
        raise ValueError(f"Invalid input path: {input_path}")

def main():
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("Starting inference script")
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    # Override config with command line arguments
    if args.output:
        config['output']['output_dir'] = args.output
        config['output']['save_results'] = True
    
    if args.scene_id:
        config['data']['scene_id'] = args.scene_id
    
    if args.visualize:
        config['output']['visualize'] = True
    
    # Check device availability
    if config['hardware']['device'] == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        config['hardware']['device'] = 'cpu'
    
    try:
        # Initialize inference engine
        engine = OnlineInferenceEngine(config)
        logger.info("Initialized inference engine")
        
        # Load input data
        scene_id = config['data'].get('scene_id', 'default_scene')
        sequence_data = load_frame_data(args.input, scene_id)
        logger.info(f"Loaded {len(sequence_data)} frames for processing")
        
        # Process sequence
        results = engine.process_sequence(sequence_data)
        
        # Print summary
        logger.info("Inference completed!")
        logger.info(f"Processed {len(results)} frames")
        
        # Print scene summary
        scene_summary = engine.get_scene_summary(scene_id)
        logger.info(f"Scene summary: {scene_summary}")
        
        # Save overall results
        if config['output']['save_results']:
            output_dir = Path(config['output']['output_dir'])
            results_file = output_dir / f"{scene_id}_results.json"
            
            # Convert tensors to lists for JSON serialization
            serializable_results = []
            for result in results:
                serializable_result = {
                    'scene_id': result['scene_id'],
                    'frame_id': result['frame_id'],
                    'processing_time': result['processing_time'],
                    'num_instances': len(torch.unique(result['predictions']['instance_ids'])),
                    'avg_instance_score': float(result['predictions']['instance_scores'].mean())
                }
                serializable_results.append(serializable_result)
            
            with open(results_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Saved results to {results_file}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Inference interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())
