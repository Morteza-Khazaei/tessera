# TESSERA Training Implementation

This directory contains the implementation for training TESSERA from scratch and adapting it for STEP-AWBH digital soil mapping.

## 📁 Directory Structure

```
tessera_training/
├── README.md                 # This file
├── TRAINING_PLAN.md         # Comprehensive training and adaptation plan
├── train.py                 # Main training script
├── config.yaml              # Training configuration
├── stepawbh_adapter.py      # STEP-AWBH adaptation (to be created)
└── utils/                   # Utility functions (to be created)
```

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Python 3.8+
# PyTorch 2.0+
# CUDA 11.8+ (for GPU training)

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pyyaml tqdm
```

### 2. Prepare Data

Follow the preprocessing pipeline in `tessera_preprocessing/` to create training data:

```bash
cd ../tessera_preprocessing

# 1. Download and process Sentinel data
bash s1_s2_downloader.sh

# 2. Stack temporal data
bash s1_s2_stacker.sh

# 3. Patchify into tiles
python dpixel_retiler.py \
    --tiff_path /path/to/roi.tif \
    --d_pixel_dir /path/to/data_processed \
    --patch_size 500 \
    --out_dir /path/to/retiled_d_pixel \
    --num_workers 16 \
    --block_size 2000
```

### 3. Configure Training

Edit `config.yaml` to set your data paths and training parameters:

```yaml
data:
  train_dirs:
    - '/path/to/retiled_d_pixel'  # Update this!
```

### 4. Start Training

```bash
# Single GPU
python train.py --config config.yaml --output_dir outputs/run1

# Multi-GPU (coming soon)
# torchrun --nproc_per_node=4 train.py --config config.yaml --output_dir outputs/run1
```

## 📊 Monitoring Training

Training logs are saved to `outputs/run1/training.log`:

```bash
# Watch training progress
tail -f outputs/run1/training.log

# TensorBoard (optional, to be implemented)
tensorboard --logdir outputs/run1/tensorboard
```

## 🔍 Understanding the Architecture

### Model Components

1. **Sentinel-2 Encoder**
   - Input: 10 spectral bands + DOY
   - Architecture: Transformer (8 layers, 8 heads)
   - Output: 512-dim representation

2. **Sentinel-1 Encoder**
   - Input: 2 polarizations (VV, VH) + DOY
   - Architecture: Same as S2
   - Output: 512-dim representation

3. **Fusion Layer**
   - Concatenates S2 + S1 → 1024-dim
   - Reduces to 128-dim embedding

4. **Projector (Training only)**
   - 6-layer MLP
   - Expands to 16384-dim for Barlow Twins

### Loss Function

**Barlow Twins Loss:**
```
L = Σ(1 - C_ii)² + λ Σ C_ij²
    i              i≠j
```

Where:
- C is the cross-correlation matrix
- First term: invariance (diagonal should be 1)
- Second term: redundancy reduction (off-diagonal should be 0)
- λ = 0.005 (default)

**Mixup Regularization:**
- Mixes two augmented views
- Adds robustness to the learned representations

## 🧪 Validation

### Compare with Pretrained Checkpoint

```python
import torch

# Load your trained model
your_checkpoint = torch.load('outputs/run1/checkpoint_epoch0_step48600.pt')

# Load pretrained model
pretrained = torch.load('tessera_infer/checkpoints/best_model_fsdp_20250427_084307.pt')

# Compare architectures
print("Your model keys:", list(your_checkpoint['model_state_dict'].keys())[:5])
print("Pretrained keys:", list(pretrained['model_state_dict'].keys())[:5])

# Compare specific layer weights
your_weight = your_checkpoint['model_state_dict']['s2_backbone.embedding.0.weight']
pretrained_weight = pretrained['model_state_dict']['_orig_mod.s2_backbone.embedding.0.weight']

print(f"Weight difference: {torch.norm(your_weight - pretrained_weight).item()}")
```

### Test on Downstream Task

```python
# Use trained embeddings for crop classification
from tessera_infer.src.models.ssl_model import MultimodalBTInferenceModel

# Load your model
checkpoint = torch.load('outputs/run1/checkpoint_epoch0_step48600.pt')
model = build_ssl_model(config, device)
model.load_state_dict(checkpoint['model_state_dict'])

# Extract embeddings
embeddings = model(s2_data, s1_data)  # (B, 128)

# Train classifier on embeddings
# ... (see downstream task examples)
```

## 🌍 STEP-AWBH Adaptation

### Overview

STEP-AWBH extends TESSERA for digital soil mapping by incorporating:

- **S**: Soil properties
- **T**: Topographic features (DEM, slope, aspect)
- **E**: Ecological factors (vegetation, land cover)
- **P**: Parent material (geology, lithology)
- **A**: Atmospheric conditions (temperature, precipitation)
- **W**: Water (soil moisture, drainage)
- **B**: Biotic factors (biomass, biodiversity)
- **H**: Human factors (land use, management)

### Implementation Steps

1. **Prepare STEP-AWBH Covariates**

```python
# Static features (T, P)
static_features = {
    'elevation': load_dem(),
    'slope': calculate_slope(dem),
    'aspect': calculate_aspect(dem),
    'geology': load_geology_map(),
    # ... more features
}

# Dynamic features (A, W, B, H)
dynamic_features = {
    'temperature': load_temperature_timeseries(),
    'precipitation': load_precipitation_timeseries(),
    'soil_moisture': load_soil_moisture_timeseries(),
    'ndvi': calculate_ndvi(s2_timeseries),
    # ... more features
}
```

2. **Adapt Model Architecture**

See `TRAINING_PLAN.md` Part 3 for detailed architecture.

3. **Train on Soil Property Prediction**

```bash
# Coming soon: stepawbh_train.py
python stepawbh_train.py \
    --pretrained outputs/run1/checkpoint_epoch0_step48600.pt \
    --soil_data /path/to/soil_measurements.csv \
    --covariates /path/to/stepawbh_covariates/ \
    --output_dir outputs/stepawbh_run1
```

## 📈 Expected Results

### Training Metrics

After replicating TESSERA training:
- **Epoch 0, Step 48,600**: Should match pretrained checkpoint
- **Validation accuracy**: ~0.79 (as in pretrained model)
- **Loss convergence**: Barlow Twins loss should decrease steadily

### Downstream Performance

Test embeddings on crop classification (Austrian dataset):
- **Weighted F1**: >0.85
- **Macro F1**: >0.80
- **Better than**: Random Forest baseline, PRESTO, comparable to GSE

## 🐛 Troubleshooting

### Out of Memory

```yaml
# Reduce batch size in config.yaml
training:
  batch_size: 128  # or 64
```

### Slow Training

```yaml
# Increase num_workers
training:
  num_workers: 16  # match your CPU cores
```

### NaN Loss

- Check data normalization
- Reduce learning rate
- Check for invalid values in input data

## 📚 References

1. **TESSERA Paper**: https://arxiv.org/abs/2506.20380
2. **Barlow Twins**: Zbontar et al. 2021, ICML
3. **STEP-AWBH**: Grunwald et al. 2011, SSSAJ
4. **Digital Soil Mapping**: McBratney et al. 2003

## 🤝 Contributing

This is a research implementation. For questions or issues:
1. Check `TRAINING_PLAN.md` for detailed documentation
2. Review training logs for debugging
3. Compare with pretrained checkpoint

## 📝 License

MIT License (same as TESSERA project)

---

**Last Updated**: 2024-12-08  
**Version**: 1.0  
**Status**: Initial Implementation
