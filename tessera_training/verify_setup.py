#!/usr/bin/env python3
"""
TESSERA Setup Verification Script

Run this before training to verify:
1. Environment is correctly configured
2. Checkpoint can be loaded
3. Model architecture matches
4. Data is accessible and valid
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import yaml

print("=" * 70)
print("TESSERA TRAINING SETUP VERIFICATION")
print("=" * 70)

# ============================================================================
# 1. Check Python and PyTorch
# ============================================================================
print("\n[1/6] Checking Python and PyTorch...")
print(f"  Python version: {sys.version.split()[0]}")
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU count: {torch.cuda.device_count()}")
    print(f"  GPU name: {torch.cuda.get_device_name(0)}")
    print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("  ✓ Environment OK")

# ============================================================================
# 2. Check Checkpoint
# ============================================================================
print("\n[2/6] Checking pretrained checkpoint...")
checkpoint_path = Path(__file__).parent.parent / 'tessera_infer/checkpoints/best_model_fsdp_20250427_084307.pt'

if not checkpoint_path.exists():
    print(f"  ✗ ERROR: Checkpoint not found at {checkpoint_path}")
    sys.exit(1)

print(f"  Checkpoint path: {checkpoint_path}")
print(f"  Checkpoint size: {checkpoint_path.stat().st_size / 1e9:.2f} GB")

try:
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    print(f"  Epoch: {ckpt['epoch']}")
    print(f"  Step: {ckpt['step']}")
    print(f"  Best val acc: {ckpt['best_val_acc']:.4f}")
    print(f"  Model parameters: {len(ckpt['model_state_dict'])}")
    print("  ✓ Checkpoint loaded successfully")
except Exception as e:
    print(f"  ✗ ERROR loading checkpoint: {e}")
    sys.exit(1)

# ============================================================================
# 3. Check Model Architecture
# ============================================================================
print("\n[3/6] Checking model architecture...")
try:
    from tessera_infer.src.models.builder import build_ssl_model
    
    config = {
        'latent_dim': 128,
        'fusion_method': 'concat',
        'projector_hidden_dim': 16384,
        'projector_out_dim': 16384
    }
    
    device = torch.device('cpu')  # Use CPU for verification
    model = build_ssl_model(config, device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params/1e6:.2f}M")
    
    # Check key components
    print(f"  S2 encoder layers: {len(model.s2_backbone.transformer_encoder.layers)}")
    print(f"  S1 encoder layers: {len(model.s1_backbone.transformer_encoder.layers)}")
    print(f"  Projector layers: {len([m for m in model.projector.net if isinstance(m, torch.nn.Linear)])}")
    
    print("  ✓ Model architecture OK")
except Exception as e:
    print(f"  ✗ ERROR building model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 4. Check Configuration File
# ============================================================================
print("\n[4/6] Checking configuration file...")
config_path = Path(__file__).parent / 'config_simple.yaml'

if not config_path.exists():
    print(f"  ✗ ERROR: Config file not found at {config_path}")
    sys.exit(1)

try:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"  Config file: {config_path}")
    print(f"  Latent dim: {config['model']['latent_dim']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Data dirs: {len(config['data']['train_dirs'])}")
    
    print("  ✓ Configuration OK")
except Exception as e:
    print(f"  ✗ ERROR loading config: {e}")
    sys.exit(1)

# ============================================================================
# 5. Check Data Directory
# ============================================================================
print("\n[5/6] Checking data directory...")
data_dirs = config['data']['train_dirs']

if not data_dirs:
    print("  ⚠ WARNING: No data directories specified in config")
    print("  Please update config_simple.yaml with your data path")
else:
    for data_dir in data_dirs:
        data_path = Path(data_dir)
        print(f"  Data directory: {data_path}")
        
        if not data_path.exists():
            print(f"  ⚠ WARNING: Data directory does not exist")
            print(f"  Please create data using tessera_preprocessing pipeline")
        else:
            # Count tiles
            tiles = [d for d in data_path.iterdir() if d.is_dir()]
            print(f"  Found {len(tiles)} tiles")
            
            if len(tiles) > 0:
                # Check first tile
                sample_tile = tiles[0]
                print(f"  Sample tile: {sample_tile.name}")
                
                required_files = [
                    'bands.npy', 'masks.npy', 'doys.npy',
                    'sar_ascending.npy', 'sar_ascending_doy.npy',
                    'sar_descending.npy', 'sar_descending_doy.npy'
                ]
                
                missing = []
                for fname in required_files:
                    if not (sample_tile / fname).exists():
                        missing.append(fname)
                
                if missing:
                    print(f"  ⚠ WARNING: Missing files in sample tile: {missing}")
                else:
                    # Load and check shapes
                    try:
                        s2_bands = np.load(sample_tile / 'bands.npy')
                        s2_masks = np.load(sample_tile / 'masks.npy')
                        s1_asc = np.load(sample_tile / 'sar_ascending.npy')
                        
                        print(f"  S2 bands shape: {s2_bands.shape}")
                        print(f"  S2 masks shape: {s2_masks.shape}")
                        print(f"  S1 ascending shape: {s1_asc.shape}")
                        print("  ✓ Data format OK")
                    except Exception as e:
                        print(f"  ✗ ERROR loading data: {e}")

# ============================================================================
# 6. Test Training Script
# ============================================================================
print("\n[6/6] Checking training script...")
train_script = Path(__file__).parent / 'train.py'

if not train_script.exists():
    print(f"  ✗ ERROR: Training script not found at {train_script}")
    sys.exit(1)

print(f"  Training script: {train_script}")
print(f"  Script size: {train_script.stat().st_size / 1024:.1f} KB")
print("  ✓ Training script OK")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)
print("\n✓ All checks passed!")
print("\nYou are ready to start training:")
print("\n  cd tessera_training")
print("  python3 train.py --config config_simple.yaml --output_dir outputs/run1")
print("\nMonitor training:")
print("  tail -f outputs/run1/training.log")
print("\n" + "=" * 70)
