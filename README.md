# HINT++

**3D point cloud semantic segmentation** — transformer backbones accelerated by optimized GPU attention kernels on the S3DIS indoor dataset.

---

## Overview

HINT++ integrates three subsystems:

| Subsystem | Role |
|---|---|
| **Pointcept** | Modular training framework for 3D point cloud perception |
| **FlashAttention** | High-performance GPU attention kernels (v2 / v3 / v4) |
| **Figures & Utilities** | Result visualization, figure generation, evaluation tools |

The primary task is **semantic segmentation** of indoor 3D scenes using transformer backbones (Point Transformer V3, Sonata, Concerto) with FlashAttention kernels for GPU-efficient attention.

---

## Repository Structure

```
HINT++/
├── Pointcept/                    # Training framework
│   ├── tools/                    # train.py, test.py, test_s3dis_6fold.py
│   ├── configs/                  # Per-dataset training configs (s3dis, scannet, sonata, …)
│   ├── pointcept/                # Core Python package
│   │   ├── datasets/             # S3DIS, ScanNet, NuScenes, Waymo, … loaders
│   │   ├── models/               # PTv3, PTv2, Sonata, Concerto, SparseUNet, …
│   │   ├── engines/              # Training + evaluation loops (DDP, AMP)
│   │   └── utils/                # Config, optimizer, scheduler, logging
│   ├── libs/                     # Custom CUDA ops (pointops KNN, ball query)
│   ├── scripts/                  # Multi-GPU shell wrappers
│   └── environment.yml           # Conda environment
│
├── flash-attention/              # FlashAttention library
│   ├── flash_attn/               # Python package (v2 API + v4 CuTeDSL kernels)
│   │   └── cute/                 # v4 — Hopper (SM90) + Blackwell (SM100)
│   ├── hopper/                   # v3 — Hopper C++ kernels (TMA/GMMA)
│   ├── csrc/flash_attn/          # v2 — CUDA C++ source (A100/RTX)
│   ├── benchmarks/               # Throughput benchmarks
│   ├── tests/                    # Correctness test suite
│   └── AI/                       # Kernel debug notes (block size tuning, race conditions)
│
├── Figures and Illustrations/    # Paper figures + generation scripts
│   ├── figures/                  # mIoU tables and class-frequency plots (PDF + PNG)
│   └── Figures Generating Codes/ # fig3_iou_nk_table.py, fig4_class_frequency.py
│
├── save_gt_files.py              # Ground-truth extraction for S3DIS Area_5 evaluation
└── CODEBASE.md                   # Detailed codebase reference
```

---

## Quick Start

### 1. Environment

```bash
cd Pointcept
conda env create -f environment.yml
conda activate pointcept
```

### 2. Dataset

Download and preprocess S3DIS (not included in this repo):

```bash
# After downloading Stanford3dDataset_v1.2, run the preprocessing script:
python pointcept/datasets/preprocessing/s3dis/preprocess_s3dis.py \
    --dataset_root /path/to/Stanford3dDataset_v1.2 \
    --output_root /path/to/s3dis-compressed
```

### 3. Train

```bash
# Single node, 4 GPUs
sh scripts/train.sh -g 4 -d s3dis -c semseg-pt-v3m1-0-base -n my_run

# Or directly:
python tools/train.py --config-file configs/s3dis/semseg-pt-v3m1-0-base.py
```

### 4. Evaluate

```bash
python tools/test.py \
    --config-file configs/s3dis/semseg-pt-v3m1-0-base.py \
    --options weight=path/to/checkpoint.pth

# 6-fold cross-validation:
python tools/test_s3dis_6fold.py
```

---

## FlashAttention Versions

| Version | Location | GPU Target | Language |
|---|---|---|---|
| v2 | `flash-attention/csrc/flash_attn/` | A100, RTX 3090/4090 (SM80) | CUDA C++ |
| v3 | `flash-attention/hopper/` | H100 (SM90) | CUDA C++ + TMA/GMMA |
| v4 | `flash-attention/flash_attn/cute/` | H100 (SM90), B200 (SM100) | Python CuTeDSL |

```python
from flash_attn import flash_attn_func, flash_attn_varlen_func

out = flash_attn_func(q, k, v, causal=True)
out = flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
```

---

## Models

| Model | Config prefix | Notes |
|---|---|---|
| Point Transformer V3 | `semseg-pt-v3m1-*` | Primary SOTA backbone |
| Sonata | `semseg-sonata-v1m1-*` | Self-supervised pre-training (CVPR 2025) |
| Concerto | `semseg-concerto-*` | 2D-3D joint learning (NeurIPS 2025) |
| SparseUNet | `semseg-spunet-*` | SpConv / MinkowskiEngine baseline |

---

## Dataset: S3DIS

13-class indoor semantic segmentation:
`ceiling · floor · wall · beam · column · window · door · table · chair · sofa · bookcase · board · clutter`

Standard evaluation: **Area 5** held-out test set, reported as mean IoU (mIoU).

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10 |
| Deep learning | PyTorch 2.5 + CUDA 12.4 |
| Sparse conv | spconv-cu124 |
| Kernel language (v4) | NVIDIA CuTeDSL 4.4+ |
| Experiment tracking | TensorBoard / W&B |
| 3D visualization | Open3D |
| Environment | Conda (`environment.yml`) |

---

## License

See `Pointcept/LICENSE` and `flash-attention/LICENSE`.
