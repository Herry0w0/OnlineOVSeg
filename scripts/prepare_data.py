#!/usr/bin/env python3
"""
Data preparation script for ScanNet dataset
"""
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import logging
from pathlib import Path
import pickle
import json
from tqdm import tqdm

from src.utils import setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare ScanNet data for training')
    parser.add_argument('--data-root', type=str, required=True,
                       help='Path to ScanNet dataset root')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for processed data')
    parser.add_argument('--scenes', type=str, nargs='+', default=None,
                       help='Specific scenes to process (default: all)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    return parser.parse_args()

def process_scene(scene_path: Path, output_dir: Path, scene_id: str):
    """Process a single scene"""
    logger = logging.getLogger(__name__)
    
    scene_output_dir = output_dir / scene_id
    scene_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if scene has required files
    color_dir = scene_path / "color"
    depth_dir = scene_path / "depth"
    instance_dir = scene_path / "instance-filt"
    
    if not color_dir.exists():
        logger.warning(f"No color directory found for scene {scene_id}")
        return False
    
    # Get frame list
    image_files = sorted(list(color_dir.glob("*.jpg")))
    logger.info(f"Found {len(image_files)} frames in scene {scene_id}")
    
    # Process frames
    frame_info = []
    for img_file in tqdm(image_files, desc=f"Processing {scene_id}"):
        frame_id = int(img_file.stem)
        
        frame_data = {
            'frame_id': frame_id,
            'image_path': str(img_file.relative_to(output_dir.parent)),
            'has_depth': (depth_dir / f"{frame_id}.png").exists(),
            'has_instance': (instance_dir / f"{frame_id}.png").exists() if instance_dir.exists() else False,
            'scene_id': scene_id
        }
        
        frame_info.append(frame_data)
    
    # Save frame info
    frame_info_file = scene_output_dir / "frame_info.json"
    with open(frame_info_file, 'w') as f:
        json.dump(frame_info, f, indent=2)
    
    # Create scene point cloud if not exists
    scene_pc_path = scene_path / "scene_point_cloud.pth"
    if not scene_pc_path.exists():
        logger.info(f"Creating scene point cloud for {scene_id}")
        
        # Try to load from mesh file
        mesh_file = scene_path / f"{scene_id}_vh_clean_2.ply"
        if mesh_file.exists():
            try:
                import open3d as o3d
                mesh = o3d.io.read_triangle_mesh(str(mesh_file))
                points = np.asarray(mesh.vertices)
                colors = np.asarray(mesh.vertex_colors)
                
                if len(colors) == 0:
                    colors = np.ones_like(points)
                
                scene_pc = torch.tensor(np.concatenate([points, colors], axis=1), dtype=torch.float32)
                torch.save(scene_pc, scene_pc_path)
                logger.info(f"Saved scene point cloud with {len(scene_pc)} points")
                
            except Exception as e:
                logger.warning(f"Failed to process mesh for {scene_id}: {e}")
                # Create dummy point cloud
                scene_pc = torch.randn(5000, 6)
                torch.save(scene_pc, scene_pc_path)
        else:
            logger.warning(f"No mesh file found for {scene_id}, creating dummy point cloud")
            scene_pc = torch.randn(5000, 6)
            torch.save(scene_pc, scene_pc_path)
    
    return True

def create_dataset_splits(data_root: Path, output_dir: Path, processed_scenes: list):
    """Create train/val/test splits"""
    logger = logging.getLogger(__name__)
    
    # Simple split: 70% train, 15% val, 15% test
    n_scenes = len(processed_scenes)
    n_train = int(0.7 * n_scenes)
    n_val = int(0.15 * n_scenes)
    
    train_scenes = processed_scenes[:n_train]
    val_scenes = processed_scenes[n_train:n_train + n_val]
    test_scenes = processed_scenes[n_train + n_val:]
    
    # Save splits
    splits = {
        'train': train_scenes,
        'val': val_scenes,
        'test': test_scenes
    }
    
    for split_name, scenes in splits.items():
        split_file = output_dir / f"{split_name}.txt"
        with open(split_file, 'w') as f:
            for scene in scenes:
                f.write(f"{scene}\n")
        logger.info(f"Created {split_name} split with {len(scenes)} scenes")
    
    # Save complete split info
    split_info_file = output_dir / "splits.json"
    with open(split_info_file, 'w') as f:
        json.dump(splits, f, indent=2)

def main():
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("Starting data preparation")
    
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find scenes to process
    scans_dir = data_root / "scans"
    if not scans_dir.exists():
        logger.error(f"Scans directory not found: {scans_dir}")
        return 1
    
    if args.scenes:
        scenes_to_process = args.scenes
    else:
        scenes_to_process = [d.name for d in scans_dir.iterdir() if d.is_dir()]
    
    logger.info(f"Found {len(scenes_to_process)} scenes to process")
    
    # Process scenes
    processed_scenes = []
    for scene_id in tqdm(scenes_to_process, desc="Processing scenes"):
        scene_path = scans_dir / scene_id
        
        if not scene_path.exists():
            logger.warning(f"Scene path not found: {scene_path}")
            continue
        
        try:
            success = process_scene(scene_path, output_dir, scene_id)
            if success:
                processed_scenes.append(scene_id)
        except Exception as e:
            logger.error(f"Failed to process scene {scene_id}: {e}")
    
    logger.info(f"Successfully processed {len(processed_scenes)} scenes")
    
    # Create dataset splits
    if processed_scenes:
        create_dataset_splits(data_root, output_dir, processed_scenes)
    
    # Save processing summary
    summary = {
        'total_scenes_found': len(scenes_to_process),
        'successfully_processed': len(processed_scenes),
        'processed_scenes': processed_scenes,
        'data_root': str(data_root),
        'output_dir': str(output_dir)
    }
    
    summary_file = output_dir / "processing_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Data preparation completed! Summary saved to {summary_file}")
    return 0

if __name__ == '__main__':
    exit(main())
