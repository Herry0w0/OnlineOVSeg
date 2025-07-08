#!/usr/bin/env python3
"""
Training script for online 3D instance segmentation
"""
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import logging
from src.utils import load_config, setup_logging
from src.training import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description='Train online 3D instance segmentation model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to training configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("Starting training script")
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    # Check device availability
    if config['hardware']['device'] == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        config['hardware']['device'] = 'cpu'
    
    try:
        # Initialize trainer
        trainer = Trainer(config)
        
        # Resume from checkpoint if specified
        if args.resume:
            if os.path.exists(args.resume):
                trainer.load_checkpoint(args.resume)
                logger.info(f"Resumed training from {args.resume}")
            else:
                logger.error(f"Checkpoint not found: {args.resume}")
                return 1
        
        # Start training
        trainer.train()
        
        logger.info("Training completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())
