"""
Training engine for online instance segmentation model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import time
from typing import Dict, Optional
import logging
from tqdm import tqdm

from ..models import OnlineInstanceSegmentationModel
from ..losses import CombinedLoss
from ..datasets import ScanNetMultiFrameDataset
from ..utils import save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)

class Trainer:
    """Training engine for the online instance segmentation model"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(config['hardware']['device'])
        
        # Initialize model
        self.model = OnlineInstanceSegmentationModel(config['model']).to(self.device)
        
        # Initialize loss function
        self.criterion = CombinedLoss(config['training']).to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize datasets and data loaders
        self.train_loader, self.val_loader = self._create_data_loaders()
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        
        # Logging
        self.writer = SummaryWriter(log_dir=os.path.join('runs', 'online_segmentation'))
        
        # Checkpointing
        self.checkpoint_dir = 'checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        logger.info("Trainer initialized successfully")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config"""
        opt_config = self.config['optimization']
        
        if opt_config['optimizer'] == 'AdamW':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=opt_config['weight_decay']
            )
        elif opt_config['optimizer'] == 'Adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=opt_config['weight_decay']
            )
        else:
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                momentum=0.9,
                weight_decay=opt_config['weight_decay']
            )
        
        return optimizer
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler"""
        opt_config = self.config['optimization']
        
        if opt_config['scheduler'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['num_epochs']
            )
        elif opt_config['scheduler'] == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            scheduler = None
        
        return scheduler
    
    def _create_data_loaders(self) -> tuple:
        """Create training and validation data loaders"""
        data_config = self.config['data']
        hw_config = self.config['hardware']
        
        # Training dataset
        train_dataset = ScanNetMultiFrameDataset(
            data_root=data_config['dataset_root'],
            split='train',
            num_frames=data_config['num_frames'],
            max_points=data_config['point_cloud_size'],
            image_size=data_config['image_size']
        )
        
        # Validation dataset
        val_dataset = ScanNetMultiFrameDataset(
            data_root=data_config['dataset_root'],
            split='val',
            num_frames=data_config['num_frames'],
            max_points=data_config['point_cloud_size'],
            image_size=data_config['image_size']
        )
        
        # Data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=hw_config['num_workers'],
            pin_memory=hw_config['pin_memory'],
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=hw_config['num_workers'],
            pin_memory=hw_config['pin_memory'],
            drop_last=False
        )
        
        logger.info(f"Created data loaders: train={len(train_loader)}, val={len(val_loader)}")
        return train_loader, val_loader
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        epoch_losses = {
            'total_loss': 0.0,
            'instance_discrimination_loss': 0.0,
            'semantic_alignment_loss': 0.0,
            'cross_frame_consistency_loss': 0.0
        }
        
        num_batches = len(self.train_loader)
        
        with tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                self.optimizer.zero_grad()
                model_output = self.model(batch)
                
                # Compute loss
                loss_dict = self.criterion(model_output, batch)
                total_loss = loss_dict['total_loss']
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                self.optimizer.step()
                
                # Accumulate losses
                for key in epoch_losses:
                    if key in loss_dict:
                        epoch_losses[key] += loss_dict[key].item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{total_loss.item():.4f}",
                    'inst': f"{loss_dict['instance_discrimination_loss'].item():.4f}",
                    'sem': f"{loss_dict['semantic_alignment_loss'].item():.4f}",
                    'cons': f"{loss_dict['cross_frame_consistency_loss'].item():.4f}"
                })
                
                # Log to tensorboard
                if batch_idx % self.config['training']['log_interval'] == 0:
                    global_step = self.current_epoch * num_batches + batch_idx
                    self._log_losses(loss_dict, global_step, 'train')
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        epoch_losses = {
            'total_loss': 0.0,
            'instance_discrimination_loss': 0.0,
            'semantic_alignment_loss': 0.0,
            'cross_frame_consistency_loss': 0.0
        }
        
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                model_output = self.model(batch)
                
                # Compute loss
                loss_dict = self.criterion(model_output, batch)
                
                # Accumulate losses
                for key in epoch_losses:
                    if key in loss_dict:
                        epoch_losses[key] += loss_dict[key].item()
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        for epoch in range(self.current_epoch, self.config['training']['num_epochs']):
            self.current_epoch = epoch
            
            # Training
            train_losses = self.train_epoch()
            
            # Validation
            val_losses = self.validate_epoch()
            
            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Logging
            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_losses['total_loss']:.4f}, "
                f"Val Loss: {val_losses['total_loss']:.4f}"
            )
            
            # Log to tensorboard
            self._log_epoch_summary(train_losses, val_losses, epoch)
            
            # Save checkpoint
            if epoch % self.config['training']['save_interval'] == 0:
                self._save_checkpoint(val_losses['total_loss'])
            
            # Save best model
            if val_losses['total_loss'] < self.best_loss:
                self.best_loss = val_losses['total_loss']
                self._save_checkpoint(val_losses['total_loss'], is_best=True)
        
        logger.info("Training completed!")
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch data to device"""
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                batch[key] = [v.to(self.device) for v in value]
        return batch
    
    def _log_losses(self, loss_dict: Dict, step: int, phase: str):
        """Log losses to tensorboard"""
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                self.writer.add_scalar(f"{phase}/{key}", value.item(), step)
            elif isinstance(value, (int, float)):
                self.writer.add_scalar(f"{phase}/{key}", value, step)
    
    def _log_epoch_summary(self, train_losses: Dict, val_losses: Dict, epoch: int):
        """Log epoch summary to tensorboard"""
        for key in train_losses:
            self.writer.add_scalars(f"epoch/{key}", {
                'train': train_losses[key],
                'val': val_losses[key]
            }, epoch)
        
        # Log learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('epoch/learning_rate', current_lr, epoch)
    
    def _save_checkpoint(self, loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f"checkpoint_epoch_{self.current_epoch}.pth"
        )
        
        save_checkpoint(
            self.model,
            self.optimizer,
            self.current_epoch,
            loss,
            checkpoint_path
        )
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            save_checkpoint(
                self.model,
                self.optimizer,
                self.current_epoch,
                loss,
                best_path
            )
            logger.info(f"Saved best model with loss: {loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        epoch, loss = load_checkpoint(checkpoint_path, self.model, self.optimizer)
        self.current_epoch = epoch + 1
        self.best_loss = loss
        logger.info(f"Loaded checkpoint from epoch {epoch} with loss {loss:.4f}")
