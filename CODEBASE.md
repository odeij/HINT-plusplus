# HINT++ Codebase Reference

> 3D point cloud scene understanding — semantic segmentation on S3DIS with transformer backbones and optimized GPU attention kernels.

---

## Project Overview

**HINT++** is a research project combining three major subsystems:

| Subsystem | Purpose |
|---|---|
| **Pointcept** | Modular training framework for 3D point cloud perception |
| **FlashAttention** | High-performance GPU attention kernels (v2 / v3 / v4) |
| **Datasets & Utilities** | S3DIS dataset, pre-trained checkpoints, evaluation tools |

The primary task is **semantic segmentation** of indoor 3D scenes using transformer-based backbones (Point Transformer V3, Sonata, Concerto) accelerated by FlashAttention.

---

## Top-Level Structure

```
HINT++/
├── Pointcept/                    # Main training framework
├── flash-attention/              # FlashAttention library (v2/v3/v4)
├── datasets/                     # Raw S3DIS dataset
├── s3dis-compressed/             # Preprocessed S3DIS (voxelized .npy files)
├── checkpoints/                  # Pre-trained model weights
├── Figures and Illustrations/    # Paper figures and generation scripts
├── save_gt_files.py              # Ground-truth extraction utility
└── CODEBASE.md                   # This file
```

---

## 1. Pointcept — Training Framework

```
Pointcept/
├── tools/                        # CLI entry points
│   ├── train.py                  # Launch training
│   ├── test.py                   # Launch evaluation / inference
│   ├── test_s3dis_6fold.py       # 6-fold cross-validation on S3DIS
│   └── create_waymo_semseg_submission.py
├── scripts/                      # Shell wrappers
│   ├── train.sh                  # Multi-GPU training wrapper
│   ├── test.sh                   # Multi-GPU evaluation wrapper
│   ├── build_image.sh            # Docker image build
│   └── create_tars.sh            # Dataset packaging
├── configs/                      # Training configuration files (100+)
│   ├── _base_/                   # Shared base configs (dataset defaults, etc.)
│   ├── s3dis/                    # S3DIS-specific training configs
│   ├── scannet/                  # ScanNet training configs
│   ├── scannet200/               # ScanNet200 configs
│   ├── scannetpp/                # ScanNet++ configs
│   ├── sonata/                   # Sonata pre-training + fine-tuning configs
│   ├── concerto/                 # Concerto 2D-3D pre-training configs
│   ├── semantic_kitti/           # SemanticKITTI (outdoor) configs
│   ├── nuscenes/                 # NuScenes configs
│   ├── waymo/                    # Waymo configs
│   ├── modelnet40/               # ModelNet40 classification configs
│   ├── matterport3d/             # Matterport3D configs
│   └── structured3d/             # Structured3D configs
├── pointcept/                    # Main Python package
│   ├── __init__.py
│   ├── datasets/                 # Dataset loaders
│   │   ├── s3dis.py              # S3DIS indoor segmentation dataset
│   │   ├── scannet.py            # ScanNet v2 dataset
│   │   ├── scannet_pair.py       # ScanNet paired-frame dataset (pre-training)
│   │   ├── scannetpp.py          # ScanNet++ dataset
│   │   ├── semantic_kitti.py     # SemanticKITTI outdoor dataset
│   │   ├── nuscenes.py           # NuScenes autonomous driving dataset
│   │   ├── waymo.py              # Waymo Open Dataset
│   │   ├── modelnet.py           # ModelNet40 classification dataset
│   │   ├── shapenet_part.py      # ShapeNet part segmentation
│   │   ├── hm3d.py               # HM3D dataset
│   │   ├── aeo.py                # AEO dataset
│   │   ├── structure3d.py        # Structured3D dataset
│   │   ├── transform.py          # Data augmentation pipeline (RandomFlip, Jitter, etc.)
│   │   ├── defaults.py           # Abstract base dataset class
│   │   ├── builder.py            # Registry-based dataset factory
│   │   ├── dataloader.py         # Custom DataLoader with point-cloud collation
│   │   ├── utils.py              # Collation helpers, batch utilities
│   │   ├── __init__.py
│   │   └── preprocessing/        # One-time preprocessing scripts per dataset
│   ├── models/                   # Model architectures
│   │   ├── point_transformer_v3/ # Point Transformer V3 — primary SOTA backbone
│   │   ├── point_transformer_v2/ # Point Transformer V2
│   │   ├── point_transformer/    # Point Transformer V1 (original)
│   │   ├── sparse_unet/          # Sparse convolution UNet (SpConv / MinkowskiEngine)
│   │   ├── sonata/               # Sonata self-supervised pre-training (CVPR 2025)
│   │   ├── concerto/             # Concerto 2D-3D joint learning (NeurIPS 2025)
│   │   ├── masked_scene_contrast/ # MSC pre-training (CVPR 2023)
│   │   ├── point_prompt_training/ # PPT multi-dataset pre-training
│   │   ├── context_aware_classifier/ # CAC segmentation head
│   │   ├── point_group/          # PointGroup instance segmentation
│   │   ├── sgiformer/            # SGIFormer instance segmentation
│   │   ├── oacnns/               # Omni-Adaptive CNNs (CVPR 2024)
│   │   ├── octformer/            # Octree Transformer backbone
│   │   ├── spvcnn/               # Sparse Voxel CNN
│   │   ├── stratified_transformer/ # Stratified attention transformer
│   │   ├── swin3d/               # Swin3D pre-trained backbone
│   │   ├── losses/               # Loss functions (cross-entropy, focal, lovász)
│   │   ├── utils/                # Checkpoint loading, weight conversion helpers
│   │   ├── default.py            # DefaultSegmentor and DefaultClassifier heads
│   │   ├── modules.py            # Shared base module classes
│   │   ├── builder.py            # Registry-based model factory
│   │   └── __init__.py
│   ├── engines/                  # Training / evaluation infrastructure
│   │   ├── train.py              # Main training loop (DDP, AMP, gradient accumulation)
│   │   ├── test.py               # Evaluation loop (sliding window, full-scene inference)
│   │   ├── launch.py             # Multi-GPU / multi-node launcher
│   │   ├── defaults.py           # Argument parsing and default config values
│   │   ├── hooks/                # Training hooks (checkpoint, validation, wandb, early stop)
│   │   └── __init__.py
│   └── utils/                    # Shared utilities
│       ├── config.py             # Hierarchical config loading and merging
│       ├── optimizer.py          # Optimizer factory (SGD, Adam, AdamW + layer-wise LR)
│       ├── scheduler.py          # LR scheduler strategies (cosine, poly, one-cycle)
│       ├── logger.py             # Structured logging to file + tensorboard
│       ├── registry.py           # Registry pattern for extensible components
│       ├── comm.py               # Distributed training communication helpers
│       ├── events.py             # Event storage for metrics tracking
│       ├── env.py                # Environment / CUDA device setup
│       ├── cache.py              # Dataset caching utilities
│       ├── misc.py               # Miscellaneous helpers
│       ├── path.py               # Path management utilities
│       ├── timer.py              # Profiling timer
│       ├── visualization.py      # 3D point cloud color visualization
│       └── __init__.py
├── libs/                         # Custom CUDA extensions (compiled separately)
│   ├── pointops/                 # KNN, ball query, interpolation CUDA ops
│   ├── pointops2/                # Updated pointops
│   ├── pointgroup_ops/           # PointGroup-specific CUDA ops
│   └── pointseg/                 # Additional segmentation ops
├── s3dis-compressed/
│   └── s3dis.tar.gz              # Compressed preprocessed S3DIS (duplicate of root)
├── environment.yml               # Conda environment (Python 3.10, PyTorch 2.5, CUDA 12.4)
├── LICENSE
└── README.md
```

### Key Entry Points

| Command | Purpose |
|---|---|
| `python tools/train.py --config-file configs/s3dis/semseg-pt-v3m1-0-base.py` | Train |
| `python tools/test.py --config-file ... --options weight=ckpt.pth` | Evaluate |
| `python tools/test_s3dis_6fold.py` | 6-fold cross-validation |
| `sh scripts/train.sh -g 4 -d s3dis -c semseg-pt-v3m1-0-base -n run_name` | Multi-GPU train |

---

## 2. FlashAttention — GPU Attention Kernels

```
flash-attention/
├── flash_attn/                   # Main Python package
│   ├── __init__.py               # Package exports
│   ├── flash_attn_interface.py   # v2 Python API (flash_attn_func, flash_attn_varlen_func)
│   ├── bert_padding.py           # Padding/unpadding utilities for variable-length sequences
│   ├── flash_attn_triton.py      # Triton fallback implementation
│   ├── flash_blocksparse_attn_interface.py # Block sparse attention interface
│   ├── flash_blocksparse_attention.py      # Block sparse implementation
│   ├── cute/                     # FlashAttention v4 — CuTeDSL-based (NEWEST)
│   │   ├── interface.py          # Public API for v4 (JIT-compiled via CuTeDSL)
│   │   ├── flash_fwd.py          # Forward kernel — Hopper (SM90)
│   │   ├── flash_fwd_sm100.py    # Forward kernel — Blackwell (SM100) with paged KV
│   │   ├── flash_bwd.py          # Backward kernel — Ampere base
│   │   ├── flash_bwd_sm90.py     # Backward kernel — Hopper (SM90)
│   │   ├── flash_bwd_sm100.py    # Backward kernel — Blackwell (SM100) 2CTA
│   │   ├── softmax.py            # Online softmax (row_max + row_sum tracking)
│   │   ├── mask.py               # Causal, local (sliding window), block sparse masking
│   │   ├── block_info.py         # Tile dimensions and block range computation
│   │   ├── tile_scheduler.py     # Tile scheduling (single, varlen-aware, persistent)
│   │   ├── copy_utils.py         # Type-converting copies and TMA (Tensor Memory Accelerator) support
│   │   ├── hopper_helpers.py     # SM90 warp-group GEMM and warp utilities
│   │   ├── blackwell_helpers.py  # SM100 UMMA-based GEMM and 2CTA support
│   │   ├── paged_kv.py           # Paged KV cache management (for inference)
│   │   ├── block_sparsity.py     # Block sparse attention support
│   │   ├── cache_utils.py        # JIT compile cache (in-memory LRU + disk)
│   │   ├── cute_dsl_utils.py     # Compilation utilities and SASS dumping
│   │   └── pyproject.toml        # Package metadata
│   ├── layers/                   # High-level attention layer wrappers
│   ├── modules/                  # nn.Module wrappers (MHA, cross-attention, etc.)
│   ├── models/                   # Pre-built model integrations
│   │   ├── llama.py              # LLaMA integration
│   │   ├── gpt.py                # GPT integration
│   │   ├── opt.py                # OPT integration
│   │   └── bert.py               # BERT integration
│   ├── ops/                      # Custom CUDA operations (fused dense, layer norm)
│   ├── losses/                   # Loss functions (cross-entropy with label smoothing)
│   └── utils/
│       ├── generation.py         # Inference / generation utilities
│       └── distributed.py        # Distributed training helpers
├── hopper/                       # FlashAttention v3 — Hopper C++ kernels (SM90)
│   ├── flash_api.cpp             # C++ kernel dispatcher / launcher
│   ├── flash_attn_interface.py   # Python interface for v3
│   ├── flash_fwd_kernel_sm90.h   # Forward kernel header (SM90)
│   ├── flash_bwd_kernel_sm90.h   # Backward kernel header (SM90)
│   ├── flash_fwd_kernel_sm80.h   # Forward kernel header (SM80, Ampere fallback)
│   ├── flash_bwd_kernel_sm80.h   # Backward kernel header (SM80)
│   ├── mainloop_fwd_sm90_tma_gmma_ws.hpp  # Hopper fwd mainloop (TMA + GMMA)
│   ├── mainloop_bwd_sm90_tma_gmma_ws.hpp  # Hopper bwd mainloop
│   ├── flash_fwd_combine_kernel.h # SplitKV result combination kernel
│   ├── epilogue_fwd.hpp          # Output epilogue (stores to global memory)
│   ├── epilogue_bwd.hpp          # Backward epilogue
│   ├── softmax.h                 # Softmax with row statistics
│   ├── mask.h                    # Attention masking
│   ├── paged_kv.h                # Paged KV cache (H100 inference)
│   ├── seqlen.h                  # Sequence length utilities
│   ├── tile_scheduler.hpp        # Tile scheduling (persistent kernel)
│   ├── heuristics.h              # Runtime kernel selection heuristics
│   ├── block.h                   # Block abstraction
│   ├── rotary.h                  # Rotary position embedding
│   ├── utils.h                   # General utilities
│   ├── setup.py                  # Build config for v3
│   ├── instantiations/           # Pre-compiled kernel instantiations
│   ├── test_flash_attn.py        # v3 tests
│   ├── test_kvcache.py           # KV cache tests
│   ├── benchmark_attn.py         # v3 benchmarks
│   └── benchmark_flash_attention_fp8.py # FP8 benchmark
├── csrc/                         # FlashAttention v2 — CUDA C++ source
│   ├── flash_attn/               # v2 core: flash_api.cpp + 120+ .cu kernel files
│   │   └── src/                  # flash_fwd_*.cu, flash_bwd_*.cu (per head-dim + dtype)
│   ├── flash_attn_ck/            # AMD ROCm / ComposableKernel backend
│   ├── layer_norm/               # Fused layer normalization CUDA kernel
│   ├── fused_dense_lib/          # Fused linear layer CUDA kernel
│   └── cutlass/                  # CUTLASS submodule (CUDA template library)
├── benchmarks/                   # Performance benchmarking scripts
│   ├── benchmark_attn.py         # Attention throughput benchmark
│   ├── benchmark_flash_attention.py # Flash vs standard attention comparison
│   ├── benchmark_causal.py       # Causal attention benchmark
│   ├── benchmark_alibi.py        # ALiBi positional bias benchmark
│   ├── benchmark_gemm.py         # GEMM kernel benchmark
│   └── bench_sm90.py             # Hopper-specific benchmarks
├── tests/                        # Test suite
│   ├── test_flash_attn.py        # v2 correctness tests
│   ├── test_flash_attn_ck.py     # ROCm CK backend tests
│   ├── test_flash_attn_triton_amd.py # Triton AMD tests
│   ├── test_rotary.py            # Rotary embedding tests
│   ├── test_util.py              # Test utilities
│   ├── cute/                     # v4 tests
│   ├── layers/                   # Layer tests
│   ├── models/                   # Model integration tests
│   ├── modules/                  # Module tests
│   ├── ops/                      # Op tests
│   └── losses/                   # Loss tests
├── AI/                           # Internal debugging notes and investigations
│   ├── CLAUDE.md                 # Architecture notes (for the flash-attention package)
│   ├── SM90_BLOCK_SIZE_TUNING.md # H100 block size tuning investigation
│   ├── DEBUG_2CTA.md             # 2CTA (SM100) debugging guide
│   ├── SM90_R2P_MASKING_SASS.md  # SASS-level masking analysis
│   ├── VARLEN_PREPROCESS_TILE_BUG.md # Variable-length tile preprocessing bug
│   ├── RACECHECK_TMA_HAZARD.md   # TMA race condition documentation
│   ├── racecheck_repro_1d_tensor.py # Race condition reproduction script
│   └── racecheck_repro_1d_bulk.py   # Bulk race condition reproduction
├── training/                     # Example training scripts (GPT / language models)
│   ├── run.py                    # Training entry point
│   ├── src/                      # Training utilities
│   ├── configs/                  # Training configs
│   ├── tests/                    # Training tests
│   ├── Dockerfile
│   └── README.md
├── examples/
│   └── inference/                # Inference usage examples
├── assets/                       # Benchmark result images and paper figures
├── third_party/
│   └── aiter/                    # AMD inference library submodule
├── setup.py                      # Build system (compiles CUDA extensions)
├── Makefile                      # Build shortcuts
├── README.md
├── usage.md                      # Usage examples
├── MANIFEST.in
├── AUTHORS
└── LICENSE
```

### FlashAttention Version Guide

| Version | Location | GPU Target | Language | Status |
|---|---|---|---|---|
| v2 | `csrc/flash_attn/` | A100, RTX 3090/4090 | CUDA C++ | Production |
| v3 | `hopper/` | H100 (SM90) | CUDA C++ + TMA/GMMA | Production |
| v4 | `flash_attn/cute/` | H100 (SM90), B200 (SM100) | Python CuTeDSL | Newest |

### Common API (all versions)
```python
from flash_attn import flash_attn_func
out = flash_attn_func(q, k, v, causal=True)               # Standard
out = flash_attn_varlen_func(q, k, v, cu_seqlens_q, ...)  # Variable-length
out = flash_attn_with_kvcache(q, k_cache, v_cache, ...)   # KV cache inference
```

---

## 3. Datasets

```
datasets/
└── Stanford3dDataset_v1.2/       # Raw S3DIS dataset
    ├── Area_1/                   # Training area
    ├── Area_2/                   # Training area
    ├── Area_3/                   # Training area
    ├── Area_4/                   # Training area
    ├── Area_5/                   # Standard evaluation area (held-out test set)
    ├── Area_6/                   # Training area
    └── [each area]/[room]/       # Per-room: Annotations/ with .txt point files

s3dis-compressed/
├── s3dis-compressed/             # Preprocessed voxelized .npy arrays
│   ├── Area_1/ … Area_6/        # Per-area preprocessed data
│   └── s3dis.tar.gz
└── git-xet.tar.gz               # Git-xet large file storage tool
```

**S3DIS label classes (13 total):**
`ceiling, floor, wall, beam, column, window, door, table, chair, sofa, bookcase, board, clutter`

---

## 4. Checkpoints

```
checkpoints/
├── pretrain-sonata-v1m1-0-base.pth    # 1.9 GB — Sonata pre-trained backbone
└── sonata/                             # Additional Sonata model variants
```

Load with:
```python
# In config: weight = "checkpoints/pretrain-sonata-v1m1-0-base.pth"
# Or via tools/test.py --options weight=checkpoints/pretrain-sonata-v1m1-0-base.pth
```

---

## 5. Figures and Illustrations

```
Figures and Illustrations/
├── figures/
│   ├── fig3_iou_nk_table.pdf / .png    # mIoU vs. parameter-count table
│   └── fig4_class_frequency.pdf / .png # Per-class performance vs. frequency
└── Figures Generating Codes/
    ├── fig3_iou_nk_table.py            # Generates fig3 (benchmark result table)
    ├── fig4_class_frequency.py         # Generates fig4 (class frequency analysis)
    └── visualize_s3dis.py              # 3D scene visualizer with predicted labels
```

---

## 6. Root-Level Utility Scripts

| File | Purpose |
|---|---|
| `save_gt_files.py` | Loads S3DIS Area_5 via Pointcept and writes `_gt.npy` ground truth files for evaluation. Runs in ~2 min, no GPU needed. |
| `CODEBASE.md` | This file. |

---

## Data Flow

```
Raw S3DIS (datasets/Stanford3dDataset_v1.2/)
        ↓ preprocessing script
Compressed .npy arrays (s3dis-compressed/)
        ↓ S3DISDataset (Pointcept/pointcept/datasets/s3dis.py)
Augmented batches (transform.py)
        ↓
Model forward (e.g. PointTransformerV3 + FlashAttention)
        ↓
Loss (cross-entropy / lovász)
        ↓
Optimizer step
        ↓
Checkpoint saved + metrics logged (TensorBoard / W&B)
        ↓
Evaluation on Area_5 → mIoU
```

---

## Tech Stack

| Layer | Technology | Version |
|---|---|---|
| Language | Python | 3.10 |
| Deep learning | PyTorch | 2.5.0 |
| GPU | CUDA + cuDNN | 12.4 / 9 |
| Sparse conv | spconv | cu124 |
| Graph ops | torch-geometric | latest |
| Kernel lang (v4) | NVIDIA CuTeDSL | 4.4+ |
| CUDA templates | CUTLASS | submodule |
| Experiment tracking | TensorBoard + W&B | latest |
| 3D visualization | open3d | latest |
| Multi-modal | CLIP | GitHub |
| Fine-tuning | PEFT (LoRA) | latest |
| Environment | Conda | environment.yml |

---

## GPU Targets

| GPU | Architecture | FlashAttention Version |
|---|---|---|
| NVIDIA A100, RTX 3090/4090 | Ampere (SM80/SM86) | v2 |
| NVIDIA H100 | Hopper (SM90) | v2, v3, v4 |
| NVIDIA B200 | Blackwell (SM100) | v4 only |
| AMD MI200/MI300 | CDNA | v2 (CK backend) |

---

## Recommended Debug Starting Points

| Goal | Files to Look At |
|---|---|
| Change training hyperparameters | `Pointcept/configs/s3dis/*.py` |
| Debug dataset loading | `Pointcept/pointcept/datasets/s3dis.py` |
| Debug model architecture | `Pointcept/pointcept/models/point_transformer_v3/` |
| Debug training loop | `Pointcept/pointcept/engines/train.py` |
| Debug evaluation metrics | `Pointcept/pointcept/engines/test.py`, `tools/test_s3dis_6fold.py` |
| Attention kernel issues | `flash-attention/flash_attn/cute/interface.py` (v4), `flash-attention/hopper/flash_attn_interface.py` (v3) |
| CUDA kernel debug guides | `flash-attention/AI/*.md` |
| Pre-training setup | `Pointcept/configs/sonata/pretrain-sonata-v1m1-0-base.py` |
| Environment / deps | `Pointcept/environment.yml`, `flash-attention/setup.py` |

---

## Naming Conventions (Config Files)

Config filenames follow the pattern: `{task}-{model}-{version}-{variant}-{dataset}-{mode}.py`

| Part | Examples |
|---|---|
| task | `semseg`, `insseg`, `pretrain` |
| model | `pt-v3m1`, `sonata-v1m1`, `spunet` |
| dataset | `s3dis`, `scannet`, `scannetpp` |
| mode | `lin` (linear probe), `dec` (decoder), `ft` (fine-tune), `ppt` (PPT head) |

Example: `semseg-sonata-v1m1-3c-s3dis-ft.py` = semantic segmentation, Sonata v1m1, variant 3c, S3DIS dataset, full fine-tune.
