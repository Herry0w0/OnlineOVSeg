#!/usr/bin/env python3
"""
Simple test script to verify the framework works
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import tempfile
from pathlib import Path

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from src.models import OnlineInstanceSegmentationModel
        from src.losses import CombinedLoss
        from src.datasets import ScanNetMultiFrameDataset
        from src.training import Trainer
        from src.inference import OnlineInferenceEngine
        from src.utils import load_config
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_model_creation():
    """Test model creation"""
    print("Testing model creation...")
    
    try:
        config = {
            'ptv3': {
                'input_dim': 6,
                'output_dim': 512,
                'num_layers': 2,
                'num_heads': 4
            },
            'adapter': {
                'input_dim': 512,
                'output_dim': 512,
                'hidden_dims': [256, 512]
            },
            'clip': {
                'model_name': 'ViT-B/32'
            }
        }
        
        model = OnlineInstanceSegmentationModel(config)
        print(f"✓ Model created successfully")
        
        # Test forward pass with dummy data
        batch = {
            'point_clouds': torch.randn(1, 3, 100, 6),  # B, num_frames, N, 6
            'images': torch.randn(1, 3, 3, 240, 320),   # B, num_frames, 3, H, W
            'instance_masks': torch.randint(0, 5, (1, 3, 240, 320)),  # B, num_frames, H, W
            'scene_point_cloud': torch.randn(1, 500, 6),
            'visibility_matrix': torch.randint(0, 2, (1, 500, 3)).bool(),
            'point_indices': [torch.randint(0, 500, (100,)) for _ in range(3)]
        }
        
        output = model(batch)
        print(f"✓ Forward pass successful")
        print(f"  Output shapes: point_features={output['point_features'].shape}")
        
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_loss_computation():
    """Test loss computation"""
    print("Testing loss computation...")
    
    try:
        from src.losses import CombinedLoss
        
        config = {
            'loss_weights': {
                'instance_discrimination': 1.0,
                'semantic_alignment': 1.0,
                'cross_frame_consistency': 0.5
            }
        }
        
        criterion = CombinedLoss(config)
        
        # Create dummy model output and batch
        model_output = {
            'point_features': torch.randn(1, 2, 50, 256),
            'adapted_features': torch.randn(1, 2, 50, 512),
            'text_features': [[torch.randn(3, 512)], [torch.randn(3, 512)]],
            'instance_info': [[{'instance_ids': [1, 2, 3], 'descriptions': {}}], 
                            [{'instance_ids': [1, 2, 3], 'descriptions': {}}]],
            'point_indices': [torch.randint(0, 200, (50,)), torch.randint(0, 200, (50,))]
        }
        
        batch = {
            'instance_masks': torch.randint(0, 4, (1, 2, 120, 160)),
            'visibility_matrix': torch.randint(0, 2, (1, 200, 2)).bool(),
        }
        
        loss_dict = criterion(model_output, batch)
        print(f"✓ Loss computation successful")
        print(f"  Total loss: {loss_dict['total_loss'].item():.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_inference_engine():
    """Test inference engine"""
    print("Testing inference engine...")
    
    try:
        # Create temporary checkpoint
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            checkpoint_path = f.name
            
        # Create dummy checkpoint
        dummy_checkpoint = {
            'epoch': 0,
            'model_state_dict': {},
            'optimizer_state_dict': {},
            'loss': 0.0
        }
        torch.save(dummy_checkpoint, checkpoint_path)
        
        config = {
            'model': {
                'checkpoint_path': checkpoint_path,
                'ptv3': {
                    'input_dim': 6,
                    'output_dim': 512,
                    'num_layers': 2,
                    'num_heads': 4
                },
                'adapter': {
                    'input_dim': 512,
                    'output_dim': 512,
                    'hidden_dims': [256, 512]
                },
                'clip': {
                    'model_name': 'ViT-B/32'
                }
            },
            'inference': {
                'consistency_threshold': 0.8,
                'feature_update_rate': 0.1,
                'online_update': True
            },
            'output': {
                'save_results': False,
                'output_dir': '/tmp'
            },
            'hardware': {
                'device': 'cpu'
            }
        }
        
        engine = OnlineInferenceEngine(config)
        
        # Test single frame processing
        points = torch.randn(100, 6)
        image = torch.randn(3, 240, 320)
        result = engine.process_frame(points, image, 'test_scene', 0)
        
        print(f"✓ Inference engine works")
        print(f"  Processing time: {result['processing_time']:.4f}s")
        
        # Cleanup
        os.unlink(checkpoint_path)
        
        return True
    except Exception as e:
        print(f"✗ Inference engine failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("Running framework tests...\n")
    
    tests = [
        test_imports,
        test_model_creation,
        test_loss_computation,
        test_inference_engine
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
        print()
    
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Framework is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == '__main__':
    exit(main())
