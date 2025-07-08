# Online 3D Instance Segmentation Framework

This project implements an online 3D instance segmentation framework inspired by ESAM, designed for processing sequential point cloud frames with multi-modal supervision.

## Features

- **Multi-frame Processing**: Handles sequential point cloud frames with temporal consistency
- **Multi-modal Supervision**: Combines point cloud features with vision-language models (CLIP + LLaVA)
- **Online Training & Inference**: Real-time processing capabilities
- **Three-part Loss Function**:
  - Instance discrimination loss (intra-instance similarity + inter-instance separation)
  - Semantic alignment loss (point features ↔ CLIP text features)
  - Cross-frame consistency loss (temporal coherence)

## Architecture Overview

```
Input: Point Cloud Sequence + RGB Images + Instance Masks
      ↓
PointTransformerV3 → Point Features
      ↓
Feature Adapter → CLIP-aligned Features  
      ↓
Multi-modal Loss:
  1. Instance Discrimination (contrastive learning)
  2. Semantic Alignment (with CLIP text encoder)
  3. Cross-frame Consistency (temporal coherence)
```

## Project Structure

```
2/
├── configs/           # Configuration files
│   ├── train_config.yaml      # Training configuration
│   └── inference_config.yaml  # Inference configuration
├── scripts/          # Main entry points
│   ├── train.py              # Training script
│   ├── inference.py          # Inference script  
│   ├── prepare_data.py       # Data preparation
│   └── evaluate.py           # Evaluation script
├── src/              # Core source code
│   ├── datasets/     # Dataset loaders (ScanNet multi-frame)
│   ├── models/       # Model architectures (PTv3, CLIP, adapter)
│   ├── losses/       # Loss functions (3-part loss)
│   ├── training/     # Training engine
│   ├── inference/    # Online inference engine
│   └── utils.py      # Utility functions
├── run_training.sh   # Training launcher script
├── run_inference.sh  # Inference launcher script
├── test_framework.py # Framework test script
├── requirements.txt  # Dependencies
└── README.md        # This file
```

## Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Test the framework**:
```bash
python test_framework.py
```

## Usage

### 1. Data Preparation

First, prepare your ScanNet data:

```bash
python scripts/prepare_data.py \
    --data-root /path/to/scannet \
    --output-dir ./data/processed \
    --scenes scene0000_00 scene0001_00
```

### 2. Training

**Method 1: Using the training script directly**
```bash
python scripts/train.py --config configs/train_config.yaml
```

**Method 2: Using the launcher script**
```bash
./run_training.sh --config configs/train_config.yaml
```

**Resume training from checkpoint**:
```bash
./run_training.sh --config configs/train_config.yaml --resume checkpoints/checkpoint_epoch_10.pth
```

### 3. Inference

**Single scene inference**:
```bash
python scripts/inference.py \
    --config configs/inference_config.yaml \
    --input /path/to/scene/directory \
    --scene-id scene0000_00 \
    --output ./results
```

**Using the launcher script**:
```bash
./run_inference.sh \
    --config configs/inference_config.yaml \
    --input /path/to/scene \
    --scene-id scene0000_00 \
    --output ./results \
    --visualize
```

### 4. Evaluation

```bash
python scripts/evaluate.py \
    --predictions ./results \
    --ground-truth /path/to/gt \
    --output evaluation_results.json
```

## Configuration

### Training Configuration (`configs/train_config.yaml`)

Key parameters to adjust:

- `data.num_frames`: Number of adjacent frames to load (default: 5)
- `data.point_cloud_size`: Maximum points per frame (default: 50000)
- `model.ptv3.output_dim`: PTv3 feature dimension (default: 512)
- `training.loss_weights`: Weights for the three loss components
- `training.batch_size`: Batch size (default: 2)
- `training.learning_rate`: Learning rate (default: 0.001)

### Inference Configuration (`configs/inference_config.yaml`)

Key parameters:

- `inference.consistency_threshold`: Temporal consistency threshold (default: 0.8)
- `inference.feature_update_rate`: Feature memory update rate (default: 0.1)
- `inference.online_update`: Enable online feature updates (default: true)

## Model Components

### 1. PointTransformerV3 Backbone
- Processes single-frame point clouds (xyz + rgb)
- Outputs per-point features
- Simplified implementation without mmcv dependencies

### 2. Feature Adapter
- Aligns PTv3 features with CLIP text feature space
- Multi-layer MLP with L2 normalization

### 3. CLIP Text Encoder
- Encodes semantic descriptions to feature vectors
- Frozen pretrained weights

### 4. LLaVA Text Generator (Mock)
- Generates instance descriptions from images and masks
- Currently uses predefined templates (can be replaced with actual LLaVA)

## Loss Functions

### 1. Instance Discrimination Loss
- **Intra-instance**: Points from same instance should have similar features
- **Inter-instance**: Points from different instances should be dissimilar
- **Compactness**: Instance features should cluster around centroids

### 2. Semantic Alignment Loss
- Point features should align with CLIP text features
- Contrastive loss between correct and incorrect alignments

### 3. Cross-frame Consistency Loss
- Overlapping points across frames should have consistent features
- Temporal smoothness regularization

## Online Inference

The inference engine maintains:
- **Feature Memory**: Historical features for each scene
- **Instance Memory**: Instance-level feature representations
- **Temporal Buffer**: Recent frame features for consistency

Features:
- Real-time frame processing
- Incremental feature updates
- Temporal consistency enforcement
- Memory management for long sequences

## Extending the Framework

### Adding New Loss Functions
1. Create new loss class in `src/losses/`
2. Add to `CombinedLoss` class
3. Update configuration files

### Adding New Models
1. Implement in `src/models/`
2. Update `OnlineInstanceSegmentationModel`
3. Modify configuration schema

### Custom Datasets
1. Inherit from base dataset class
2. Implement required methods
3. Update data loading pipeline

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or point cloud size
2. **Slow training**: Reduce number of frames or use smaller model
3. **Poor temporal consistency**: Adjust consistency threshold
4. **Import errors**: Ensure all dependencies are installed

### Performance Tips

- Use mixed precision training with `torch.cuda.amp`
- Enable pin_memory for faster data loading
- Use multiple workers for data loading
- Cache preprocessed scene point clouds

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{online_instance_segmentation_framework,
  title={Online 3D Instance Segmentation Framework},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/online-3d-segmentation}}
}
```
