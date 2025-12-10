#!/usr/bin/env python3
"""
TESSERA Training Script
Implements Barlow Twins self-supervised learning for satellite time series

Usage:
    python train.py --config config.yaml
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import os
import sys
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from tessera_infer.src.models.builder import build_ssl_model
from tessera_infer.src.models.ssl_model import MultimodalBTModel


# ============================================================================
# Loss Functions
# ============================================================================

class BarlowTwinsLoss(nn.Module):
    """
    Barlow Twins loss for self-supervised learning
    
    Reference: Zbontar et al. 2021 - Barlow Twins: Self-Supervised Learning 
               via Redundancy Reduction
    """
    def __init__(self, lambda_param=0.005, scale_loss=0.024):
        super().__init__()
        self.lambda_param = lambda_param
        self.scale_loss = scale_loss
    
    def forward(self, z1, z2):
        """
        Args:
            z1: Projections from first augmentation (B, D)
            z2: Projections from second augmentation (B, D)
        
        Returns:
            loss: Barlow Twins loss value
            metrics: Dictionary of loss components
        """
        batch_size = z1.shape[0]
        feature_dim = z1.shape[1]
        
        # Normalize representations along batch dimension
        z1_norm = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + 1e-6)
        z2_norm = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + 1e-6)
        
        # Cross-correlation matrix
        c = torch.mm(z1_norm.T, z2_norm) / batch_size
        
        # Invariance term: diagonal should be 1
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        
        # Redundancy reduction: off-diagonal should be 0
        off_diag_mask = ~torch.eye(feature_dim, dtype=torch.bool, device=c.device)
        off_diag = c[off_diag_mask].pow_(2).sum()
        
        loss = on_diag + self.lambda_param * off_diag
        loss = loss * self.scale_loss
        
        metrics = {
            'on_diag': on_diag.item(),
            'off_diag': off_diag.item(),
            'total': loss.item()
        }
        
        return loss, metrics


class MixupRegularization(nn.Module):
    """Mixup regularization for enhanced robustness"""
    def __init__(self, alpha=0.2):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, z1, z2):
        """Apply mixup between two augmented views"""
        batch_size = z1.shape[0]
        
        # Sample mixing coefficient
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((batch_size, 1)).to(z1.device)
        
        # Mix representations
        z_mixed = lam * z1 + (1 - lam) * z2
        
        return z_mixed


class CombinedLoss(nn.Module):
    """Combined Barlow Twins + Mixup loss"""
    def __init__(self, lambda_bt=0.005, alpha_mix=0.2, mix_weight=0.1):
        super().__init__()
        self.bt_loss = BarlowTwinsLoss(lambda_param=lambda_bt)
        self.mixup = MixupRegularization(alpha=alpha_mix)
        self.mix_weight = mix_weight
    
    def forward(self, z1, z2, z3=None):
        """
        Args:
            z1, z2: Two augmented views
            z3: Optional third view for additional regularization
        """
        # Barlow Twins loss
        bt_loss, bt_metrics = self.bt_loss(z1, z2)
        
        # Mixup regularization
        if z3 is not None:
            z_mixed = self.mixup(z1, z2)
            mix_loss, mix_metrics = self.bt_loss(z_mixed, z3)
            total_loss = bt_loss + self.mix_weight * mix_loss
            
            metrics = {
                'bt_loss': bt_loss.item(),
                'mix_loss': mix_loss.item(),
                'total_loss': total_loss.item(),
                **{f'bt_{k}': v for k, v in bt_metrics.items()}
            }
        else:
            total_loss = bt_loss
            metrics = {
                'bt_loss': bt_loss.item(),
                'total_loss': total_loss.item(),
                **{f'bt_{k}': v for k, v in bt_metrics.items()}
            }
        
        return total_loss, metrics


# ============================================================================
# Data Augmentation
# ============================================================================

class TemporalAugmentation:
    """Temporal augmentation for satellite time series"""
    def __init__(self, sample_length=40, num_augmentations=3):
        self.sample_length = sample_length
        self.num_augmentations = num_augmentations
    
    def __call__(self, s2_bands, s2_masks, s2_doys, s1_bands, s1_doys):
        """
        Create multiple augmented views by random temporal sampling
        
        Args:
            s2_bands: (T_s2, 10) Sentinel-2 bands
            s2_masks: (T_s2,) validity masks
            s2_doys: (T_s2,) day of year
            s1_bands: (T_s1, 2) Sentinel-1 bands
            s1_doys: (T_s1,) day of year
        
        Returns:
            List of augmented samples
        """
        augmentations = []
        
        for _ in range(self.num_augmentations):
            # Sample S2
            s2_valid_idx = np.where(s2_masks > 0)[0]
            if len(s2_valid_idx) >= self.sample_length:
                s2_sampled_idx = np.random.choice(s2_valid_idx, 
                                                   self.sample_length, 
                                                   replace=False)
            else:
                # Sample with replacement if insufficient observations
                s2_sampled_idx = np.random.choice(s2_valid_idx, 
                                                   self.sample_length, 
                                                   replace=True)
            
            s2_sampled_idx = np.sort(s2_sampled_idx)
            
            # Sample S1
            s1_valid_idx = np.where(np.any(s1_bands != 0, axis=-1))[0]
            if len(s1_valid_idx) >= self.sample_length:
                s1_sampled_idx = np.random.choice(s1_valid_idx, 
                                                   self.sample_length, 
                                                   replace=False)
            else:
                s1_sampled_idx = np.random.choice(s1_valid_idx, 
                                                   self.sample_length, 
                                                   replace=True)
            
            s1_sampled_idx = np.sort(s1_sampled_idx)
            
            # Create augmented sample
            aug_sample = {
                's2': np.concatenate([
                    s2_bands[s2_sampled_idx],
                    s2_doys[s2_sampled_idx].reshape(-1, 1)
                ], axis=1).astype(np.float32),  # (40, 11)
                's1': np.concatenate([
                    s1_bands[s1_sampled_idx],
                    s1_doys[s1_sampled_idx].reshape(-1, 1)
                ], axis=1).astype(np.float32)   # (40, 3)
            }
            
            augmentations.append(aug_sample)
        
        return augmentations


# ============================================================================
# Dataset
# ============================================================================

class TESSERATrainingDataset(Dataset):
    """Dataset for TESSERA self-supervised training"""
    def __init__(self, 
                 tile_dirs,
                 sample_length=40,
                 num_augmentations=3,
                 min_valid_timesteps=10):
        """
        Args:
            tile_dirs: List of directories containing preprocessed tiles
            sample_length: Number of timesteps to sample
            num_augmentations: Number of augmented views per pixel
            min_valid_timesteps: Minimum valid observations required
        """
        self.tile_dirs = tile_dirs
        self.sample_length = sample_length
        self.num_augmentations = num_augmentations
        self.min_valid_timesteps = min_valid_timesteps
        
        self.augmentor = TemporalAugmentation(sample_length, num_augmentations)
        
        # Load and index all valid pixels
        logging.info('Building pixel index...')
        self.pixel_index = self._build_pixel_index()
        logging.info(f'Found {len(self.pixel_index)} valid pixels')
        
        # Cache for memory-mapped arrays (will be populated per worker)
        self._mmap_cache = {}
    
    def _build_pixel_index(self):
        """Build index of all valid pixels across tiles using memory mapping"""
        pixel_index = []
        
        for tile_dir in tqdm(self.tile_dirs, desc='Indexing tiles'):
            try:
                # Use memory mapping to avoid loading entire arrays
                s2_bands = np.load(os.path.join(tile_dir, 'bands.npy'), mmap_mode='r')
                s2_masks = np.load(os.path.join(tile_dir, 'masks.npy'), mmap_mode='r')
                s1_asc = np.load(os.path.join(tile_dir, 'sar_ascending.npy'), mmap_mode='r')
                s1_desc = np.load(os.path.join(tile_dir, 'sar_descending.npy'), mmap_mode='r')
                
                T, H, W, _ = s2_bands.shape
                
                # Find valid pixels
                for i in range(H):
                    for j in range(W):
                        s2_valid = np.sum(s2_masks[:, i, j])
                        s1_valid = (np.sum(np.any(s1_asc[:, i, j, :] != 0, axis=-1)) +
                                   np.sum(np.any(s1_desc[:, i, j, :] != 0, axis=-1)))
                        
                        # Check if S2 bands have non-zero values
                        s2_nonzero = np.any(s2_bands[:, i, j, :] != 0)
                        
                        if (s2_nonzero and 
                            s2_valid >= self.min_valid_timesteps and 
                            s1_valid >= self.min_valid_timesteps):
                            pixel_index.append((tile_dir, i, j))
                
                # Clean up memory maps
                del s2_bands, s2_masks, s1_asc, s1_desc
                
            except Exception as e:
                logging.warning(f'Error processing tile {tile_dir}: {e}')
                continue
        
        return pixel_index
    
    def __len__(self):
        return len(self.pixel_index)
    
    def _get_mmap(self, tile_dir, filename):
        """Get or create memory-mapped array for a file"""
        key = (tile_dir, filename)
        if key not in self._mmap_cache:
            filepath = os.path.join(tile_dir, filename)
            self._mmap_cache[key] = np.load(filepath, mmap_mode='r')
        return self._mmap_cache[key]
    
    def __getitem__(self, idx):
        tile_dir, i, j = self.pixel_index[idx]
        
        # Load pixel data using memory mapping
        s2_bands = self._get_mmap(tile_dir, 'bands.npy')[:, i, j, :].copy()
        s2_masks = self._get_mmap(tile_dir, 'masks.npy')[:, i, j].copy()
        s2_doys = self._get_mmap(tile_dir, 'doys.npy')[:].copy()
        
        s1_asc = self._get_mmap(tile_dir, 'sar_ascending.npy')[:, i, j, :].copy()
        s1_desc = self._get_mmap(tile_dir, 'sar_descending.npy')[:, i, j, :].copy()
        s1_asc_doy = self._get_mmap(tile_dir, 'sar_ascending_doy.npy')[:].copy()
        s1_desc_doy = self._get_mmap(tile_dir, 'sar_descending_doy.npy')[:].copy()
        
        # Combine S1 ascending and descending
        s1_bands = np.concatenate([s1_asc, s1_desc], axis=0)
        s1_doys = np.concatenate([s1_asc_doy, s1_desc_doy], axis=0)
        
        # Create augmentations
        augmentations = self.augmentor(s2_bands, s2_masks, s2_doys, 
                                       s1_bands, s1_doys)
        
        # Convert to tensors
        aug_tensors = []
        for aug in augmentations:
            aug_tensors.append({
                's2': torch.FloatTensor(aug['s2']),
                's1': torch.FloatTensor(aug['s1'])
            })
        
        return aug_tensors


# ============================================================================
# Collate Function
# ============================================================================

def collate_augmentations(batch):
    """
    Custom collate function for augmentations
    
    Args:
        batch: List of augmentation lists from dataset
               Each item is a list of 3 dicts: [{'s2': tensor, 's1': tensor}, ...]
    
    Returns:
        List of 3 batched augmentations
    """
    # batch is a list of length batch_size
    # Each element is a list of 3 augmentations
    # We need to reorganize to get 3 lists of batch_size augmentations
    
    num_augmentations = len(batch[0])  # Should be 3
    
    batched_augmentations = []
    for aug_idx in range(num_augmentations):
        # Collect all augmentations at this index across the batch
        s2_list = [item[aug_idx]['s2'] for item in batch]
        s1_list = [item[aug_idx]['s1'] for item in batch]
        
        batched_augmentations.append({
            's2': torch.stack(s2_list),
            's1': torch.stack(s1_list)
        })
    
    return batched_augmentations


# ============================================================================
# Training
# ============================================================================

def train_epoch(model, train_loader, optimizer, scheduler, criterion, device, epoch, log_interval=100):
    """Train for one epoch"""
    model.train()
    
    epoch_loss = 0.0
    epoch_metrics = {}
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, augmentations_batch in enumerate(pbar):
        # augmentations_batch is already a list of 3 batched augmentations from collate_fn
        # Each element is {'s2': (B, 40, 11), 's1': (B, 40, 3)}
        aug1 = {
            's2': augmentations_batch[0]['s2'].to(device),
            's1': augmentations_batch[0]['s1'].to(device)
        }
        aug2 = {
            's2': augmentations_batch[1]['s2'].to(device),
            's1': augmentations_batch[1]['s1'].to(device)
        }
        aug3 = {
            's2': augmentations_batch[2]['s2'].to(device),
            's1': augmentations_batch[2]['s1'].to(device)
        }
        
        # Forward pass
        z1, repr1 = model(aug1['s2'], aug1['s1'])
        z2, repr2 = model(aug2['s2'], aug2['s1'])
        z3, repr3 = model(aug3['s2'], aug3['s1'])
        
        # Compute loss
        loss, metrics = criterion(z1, z2, z3)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Logging
        epoch_loss += loss.item()
        for k, v in metrics.items():
            epoch_metrics[k] = epoch_metrics.get(k, 0) + v
        
        if batch_idx % log_interval == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })
    
    # Average metrics
    num_batches = len(train_loader)
    epoch_loss /= num_batches
    for k in epoch_metrics:
        epoch_metrics[k] /= num_batches
    
    return epoch_loss, epoch_metrics


def main(args):
    # Create output directory FIRST
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    logging.info(f'Config: {config}')
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Build model
    logging.info('Building model...')
    model = build_ssl_model(config['model'], device)
    logging.info(f'Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
    
    # Dataset
    logging.info('Loading dataset...')
    tile_dirs = []
    for data_dir in config['data']['train_dirs']:
        tile_dirs.extend([os.path.join(data_dir, d) for d in os.listdir(data_dir) 
                         if os.path.isdir(os.path.join(data_dir, d))])
    
    train_dataset = TESSERATrainingDataset(
        tile_dirs=tile_dirs,
        sample_length=config['data']['sample_length'],
        num_augmentations=config['data']['num_augmentations'],
        min_valid_timesteps=config['data']['min_valid_timesteps']
    )
    
    # Use fewer workers to avoid deadlock, enable persistent workers
    num_workers = min(4, config['training']['num_workers'])  # Limit to 4 workers
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=collate_augmentations
    )
    
    logging.info(f'DataLoader: batch_size={config["training"]["batch_size"]}, num_workers={num_workers}')
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    # Scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader) * config['training']['num_epochs'],
        eta_min=config['training']['min_lr']
    )
    
    # Loss
    criterion = CombinedLoss(
        lambda_bt=config['loss']['lambda_bt'],
        alpha_mix=config['loss']['alpha_mix'],
        mix_weight=config['loss']['mix_weight']
    )
    
    # Training loop
    logging.info('Starting training...')
    global_step = 0
    
    for epoch in range(config['training']['num_epochs']):
        epoch_loss, epoch_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, 
            device, epoch, config['training']['log_interval']
        )
        
        # Log epoch summary
        logging.info(f'Epoch {epoch} Summary:')
        logging.info(f'  Loss: {epoch_loss:.4f}')
        for k, v in epoch_metrics.items():
            logging.info(f'  {k}: {v:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % config['training']['save_interval'] == 0:
            checkpoint = {
                'epoch': epoch,
                'step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': epoch_loss,
                'metrics': epoch_metrics,
                'config': config
            }
            
            checkpoint_path = os.path.join(
                args.output_dir,
                f'checkpoint_epoch{epoch}_step{global_step}.pt'
            )
            torch.save(checkpoint, checkpoint_path)
            logging.info(f'Checkpoint saved: {checkpoint_path}')
        
        global_step += len(train_loader)
    
    logging.info('Training complete!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train TESSERA model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    
    args = parser.parse_args()
    main(args)
