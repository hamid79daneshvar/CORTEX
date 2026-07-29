# CORTEX: Cooperative Occlusion-Resilient Trajectory Execution via Request-Aware V2I Fusion

[![Journal: IEEE Access](https://img.shields.io/badge/IEEE%20Access-Published-blue.svg)](https://ieeeaccess.ieee.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![Benchmark: V2XVerse](https://img.shields.io/badge/Benchmark-V2XVerse-green.svg)](https://huggingface.co/datasets/gjliu/V2Xverse/tree/main)
[![Python: 3.7+](https://img.shields.io/badge/Python-3.7%2B-brightgreen.svg)](https://www.python.org/)

Official PyTorch implementation of the research paper:

> **CORTEX: Cooperative Occlusion-Resilient Trajectory Execution via Request-Aware V2I Fusion**  
> **Authors:** Hamid Daneshvar, Masoud Masih-Tehrani, and Morteza Mollajafari  
> *School of Automotive Engineering, Iran University of Science and Technology (IUST)*  
> **Journal:** *IEEE Access* | **DOI:** 10.1109/ACCESS.2024 (Manuscript ID: `Access-2025-56941`)

---

## 📖 Table of Contents
- [Executive Overview](#-executive-overview)
- [Key Architectural Contributions](#-key-architectural-contributions)
- [System Architecture](#-system-architecture)
- [Project Directory Structure](#-project-directory-structure)
- [Environment Setup & Installation](#-environment-setup--installation)
- [Dataset Preparation](#-dataset-preparation)
- [Training Pipeline](#-training-pipeline)
- [Evaluation & Benchmarking Scripts](#-evaluation--benchmarking-scripts)
- [Main Experimental Results](#-main-experimental-results)
- [Cross-Platform & Windows Compatibility Notes](#-cross-platform--windows-compatibility-notes)
- [Citation](#-citation)
- [License & Acknowledgements](#-license--acknowledgements)

---

## 💡 Executive Overview

Traditional end-to-end (E2E) learning architectures for autonomous driving experience severe **epistemic uncertainty** and passive **"freezing robot"** behaviors in dense urban environments due to non-line-of-sight (NLOS) occlusions. While Vehicle-to-Infrastructure (V2I) communication can expand the perceptual horizon, existing cooperative pipelines either saturate wireless channels via indiscriminate feature broadcasting or terminate prematurely at perception metrics (3D IoU) without directly conditioning downstream vehicle control.

**CORTEX** bridges this critical gap by unifying request-aware spatial fusion, velocity-yaw-rate motion latency correction, and a topography-preserving spatial convolutional control head directly mapped to physical actuators within a non-truncated **13,960-channel control input space**.

```text
[Raw Ego/RSU LiDAR] ──► [Pillar BEV Backbone] ──► [Request-Aware Spatial Mask]
                                                        │
[Velocity & Yaw Rate] ──► [Kinematic Latency Corrector] ┼─► [Hard-Masked Cross-Attention]
                                                        │
[13,960-Ch Control Head] ◄── [Topography Downsampler] ◄─┴─► [Autoregressive GRU Decoder]
          │
          └──► Actuators: Throttle [0,1] | Steering [-1,1] | Brake [0,1]
```

---

## ✨ Key Architectural Contributions

1. **Request-Aware Transmission Pruning:** Projects feedforward path trajectories from an egocentric coarse head to query infrastructure Roadside Units (RSUs) exclusively along the driving corridor, achieving up to **54.9% volumetric bandwidth relief**.
2. **Kinematic Latency Corrector Net (`LatencyCorrector`):** Utilizes real-time egocentric linear velocities ($v_x, v_y$) and instantaneous yaw rate ($\omega$) via 2D affine grid warping ($A_{	ext{lat}}$) and residual flow networks to neutralize asynchronous transmission lags up to **500 ms** (with only 7.4 mm tracking drift).
3. **Topography-Preserving Spatial Control Head:** Bypasses early Global Average Pooling (GAP) layers using cascading strided convolutions ($384 	imes 96 	imes 288 
ightarrow 32 	imes 12 	imes 36 
ightarrow 13,824$), preserving localized obstacle proximity boundaries within a non-truncated **13,960-channel** multi-modal control vector.
4. **Sub-Grid Pose Noise Immunity:** Sub-grid pooling quantization creates a $0.5	ext{m} 	imes 0.5	ext{m}$ token receptive field, rendering policy execution completely invariant to Gaussian localization pose drift ($\sigma \le 0.5	ext{m}$).
5. **Jitter-Free Control Synthesis:** Regularized by an explicit second-order derivative kinematic consistency loss ($\mathcal{L}_{	ext{consistency}}$), reducing global steering jitter (RMS Yaw Rate) by **74.1%** ($12.83^\circ/	ext{s}$ vs. $49.66^\circ/	ext{s}$).

---

## 🏗️ System Architecture

![CORTEX Architecture](assets/architecture.png)

### Mathematical Formulation Summary

$$\mathcal{L}_{	ext{total}} = lpha \mathcal{L}_{	ext{wp}} + eta \mathcal{L}_{	ext{ctrl}} + \gamma \mathcal{L}_{	ext{coarse}} + \lambda \mathcal{L}_{	ext{consistency}}$$

Where:
- $\mathcal{L}_{	ext{wp}}$: Waypoint trajectory regression ($L_1$ norm over look-ahead horizon $P=4$).
- $\mathcal{L}_{	ext{ctrl}}$: Actuator command loss over continuous action vector $a_t = [	ext{throttle}, 	ext{steer}, 	ext{brake}]^T$.
- $\mathcal{L}_{	ext{coarse}}$: Auxiliary supervisory signal for the feedforward query corridor generator.
- $\mathcal{L}_{	ext{consistency}}$: Relative displacement derivative smoother enforcing second-order kinematic continuity.

---

## 📂 Project Directory Structure

```text
CORTEX/
├── assets/                       # Architectural diagrams, qualitative GIFs, and documentation figures
│   ├── architecture.png
│   ├── demo_left.gif
│   └── demo_right.gif
├── cortex/                       # Proposed CORTEX Core Architecture (Ours)
│   ├── __init__.py
│   ├── backbone/
│   │   ├── pillar_vfe.py        # Pillar Feature Network (PFN) with Group Normalization
│   │   ├── pointpillar_scatter.py
│   │   └── resnet_bev.py        # Cascading multi-scale ResNet BEV encoder
│   ├── fusion/
│   │   ├── request_mask.py      # Coarse trajectory head & 2D Gaussian corridor mask
│   │   ├── latency_corrector.py # Affine grid warping & residual flow latency compensation
│   │   └── masked_attention.py  # Hard-masked scaled dot-product cross-attention
│   ├── heads/
│   │   ├── spatial_control_head.py # Non-truncated 13,960-ch spatial downsampling head
│   │   └── gru_decoder.py       # Autoregressive recurrent waypoint decoder
│   ├── dataset.py               # Cross-platform V2XVerse dataloader (Windows/Linux)
│   ├── loss.py                  # Multi-objective supervisory loss pipeline
│   ├── model.py                 # Unified CORTEX computational graph wrapper
│   └── train.py                 # Multi-GPU training script with AMP support
├── baselines/                    # Benchmark Baselines for Comparative Analysis
│   ├── tcp_reproduced/          # Monocular TCP baseline implementation
│   ├── codriving/               # Dual-map handshake request baseline
│   └── uniad_adapter/           # Centralized query planner adapter
├── evaluation/                   # Evaluation Suite & Telemetry Analysis
│   ├── evaluate_cortex.py       # Open-loop replication fidelity & ADE/FDE metrics
│   ├── stress_test_latency.py   # Stochastic V2I transmission lag injector (0 - 500 ms)
│   ├── stress_test_noise.py     # Gaussian pose uncertainty stress testing (sigma = 0 - 0.5m)
│   ├── telemetry_audit.py       # Volumetric network bandwidth & compression logger
│   └── plot_results.py          # Publication-grade kinematic & trajectory plotting
├── dataset/                      # V2XVerse Dataset directory link/symlink
│   └── weather-0/
│       └── data/
├── gen_index.py                  # Cross-platform dataset indexing script
├── requirements.txt              # Standardized Python dependencies
├── setup.py                      # Package installation setup script
└── README.md                     # Technical Documentation
```

---

## 🛠️ Environment Setup & Installation

### System Requirements
- **Operating System:** Linux (Ubuntu 20.04/22.04) or Windows 10/11 (fully supported)
- **Python:** Version 3.7, 3.8, or 3.9
- **GPU:** NVIDIA RTX 3090, RTX 4090, or A100 ($\ge 24	ext{ GB}$ VRAM recommended)
- **CUDA Toolkit:** 11.3 / 11.7 / 11.8

### Step-by-Step Installation

```bash
# 1. Clone the official repository
git clone https://github.com/hamid79daneshvar/CORTEX.git
cd CORTEX

# 2. Create and activate isolated Conda environment
conda create -n cortex python=3.7 -y
conda activate cortex

# 3. Install PyTorch with CUDA 11.3 support
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# 4. Install auxiliary dependencies
pip install -r requirements.txt

# 5. Install CORTEX package in editable mode
pip install -e .
```

---

## 💾 Dataset Preparation

The experiments use the high-fidelity **V2XVerse** benchmark dataset built upon CARLA.

1. **Download Dataset:**  
   Fetch raw point cloud splits from HuggingFace: [V2XVerse Repository](https://huggingface.co/datasets/gjliu/V2Xverse/tree/main).

2. **Directory Alignment:**  
   Organize the dataset directory as follows:

```text
CORTEX/dataset/
└── weather-0/
    └── data/
        ├── routes_town01_...
        ├── routes_town02_...
        ├── routes_town03_...
        ├── routes_town04_...
        ├── routes_town05_...   # Primary Validation / Test Split
        ├── routes_town06_...
        ├── routes_town07_...   # Validation Split
        └── routes_town10_...   # Validation Split
```

3. **Generate Preprocessed Spatial Index:**  
   Execute the cross-platform dataset indexer to cache geodetic coordinates and pillar boundaries:

```bash
python gen_index.py --root dataset --output dataset/index_cache.json
```

---

## 🚀 Training Pipeline

### 1. Train Full CORTEX Model (V2I Request-Aware Fusion)

```bash
python cortex/train.py     --id cortex_v2i_run1     --dataset_root dataset     --batch_size 16     --lr 1e-4     --gpus 1     --epochs 60     --train_towns town01 town02 town03 town04 town06     --val_towns town07 town10
```

### 2. Train CORTEX Ego-Only Baseline (Communication Blackout Mode)

```bash
python cortex/train.py     --id cortex_ego_only     --dataset_root dataset     --disable_v2i     --batch_size 16     --gpus 1
```

### 3. Train Monocular TCP Baseline

```bash
python baselines/tcp_reproduced/train_baseline.py     --id tcp_vision_baseline     --dataset_root dataset     --batch_size 32     --gpus 1
```

---

## 📈 Evaluation & Benchmarking Scripts

### 1. Standard Open-Loop Replication Fidelity (Town05)

```bash
python evaluation/evaluate_cortex.py     --checkpoint_path logs/cortex_v2i_run1/checkpoints/best_model.ckpt     --dataset_root dataset     --towns town05     --output_file results_cortex.json
```

### 2. Network Transmission Latency Stress Test ($\Delta t = 0 
ightarrow 500	ext{ ms}$)

```bash
python evaluation/stress_test_latency.py     --checkpoint_path logs/cortex_v2i_run1/checkpoints/best_model.ckpt     --dataset_root dataset     --delays_ms 0 100 200 300 400 500     --output_file latency_stress_results.json
```

### 3. Localization Pose Uncertainty Test ($\sigma = 0.0 
ightarrow 0.5	ext{ m}$)

```bash
python evaluation/stress_test_noise.py     --checkpoint_path logs/cortex_v2i_run1/checkpoints/best_model.ckpt     --dataset_root dataset     --noise_stds 0.0 0.2 0.5     --output_file pose_noise_results.json
```

### 4. Continuous Network Telemetry & Bandwidth Audit

```bash
python evaluation/telemetry_audit.py     --checkpoint_path logs/cortex_v2i_run1/checkpoints/best_model.ckpt     --dataset_root dataset     --scenarios left_turn_complex right_turn_occluded
```

### 5. Generate Publication Figures

```bash
python evaluation/plot_results.py     --cortex_json results_cortex.json     --latency_json latency_stress_results.json     --output_dir assets/plots
```

---

## 📊 Main Experimental Results

### Table 1: State-of-the-Art Benchmark Comparison on CARLA Town05 (Val Split)

| Paradigm / Model | Modality | Look-Ahead Horizon ($P$) | Planning ADE (m) ↓ | Planning FDE (m) ↓ |
| :--- | :--- | :---: | :---: | :---: |
| **No Collaboration (Ego)** | Single-Agent LiDAR | 10 steps (2.0s) | 0.636 | 1.460 |
| **Late Fusion** | Bounding Box V2X | 10 steps (2.0s) | 0.631 | 1.454 |
| **F-Cooper** | Max-Pooling V2X | 10 steps (2.0s) | 0.627 | 1.446 |
| **V2X-ViT** | Vision Transformer | 10 steps (2.0s) | 0.629 | 1.447 |
| **CoopDet3D** | Intermediate Fusion | 10 steps (2.0s) | 0.623 | 1.439 |
| **CoDriving (Full)** | Request-Aware V2X | 10 steps (2.0s) | 0.619 | 1.413 |
| **UniAD** | Centralized Query Planner | 10 steps (2.0s) | 1.080 | 1.470 |
| **TCP Baseline** | Single-Agent Camera | 4 steps (0.8s) | 0.680 | 1.496 |
| **CORTEX Ego-Only (Ours)** | Isolated Dual-LiDAR | 4 steps (0.8s) | **0.325** | **0.582** |
| **CORTEX V2I-Sync (Ours)** | Request-Aware V2I | 4 steps (0.8s) | **0.315** | **0.561** |

---

### Table 2: Spatio-Temporal Robustness Across Stochastic Network Transmission Lags ($\Delta t$)

| Communication Matrix State | Channel Delay ($\Delta t$) | Global ADE (m) ↓ | Global FDE (m) ↓ | Lateral Dev (m) ↓ | Heading Error (deg) ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CORTEX V2I-Sync (Ideal)** | $0	ext{ ms}$ | **0.3157** | **0.5614** | **0.0474** | **24.56** |
| **CORTEX Delayed State** | $100	ext{ ms}$ | 0.3178 | 0.5640 | 0.0499 | 24.89 |
| **CORTEX Delayed State** | $200	ext{ ms}$ | 0.3183 | 0.5645 | 0.0477 | 24.62 |
| **CORTEX Delayed State** | $300	ext{ ms}$ | 0.3202 | 0.5673 | 0.0481 | 24.70 |
| **CORTEX Delayed State** | $400	ext{ ms}$ | 0.3218 | 0.5677 | 0.0483 | 24.87 |
| **CORTEX Delayed State** | $500	ext{ ms}$ | **0.3252** | **0.5730** | **0.0484** | **24.88** |
| **CORTEX Ego-Only** | Blackout ($0\%$) | 0.3258 | 0.5827 | 0.0449 | 26.45 |

*Key Takeaway:* Scaling transmission delay to a critical half-second block ($500	ext{ ms}$) causes only **7.4 mm** drift in global ADE, proving total latency immunity via motion-compensated affine grid warping.

---

### Table 3: Continuous Actuation Smoothness & Steering Jitter Metrics

| Operating Communication Mode | Global RMS Jerk ($	ext{m/s}^3$) ↓ | Global RMS Steer Rate ($	ext{rad/s}$) ↓ | Global RMS Steer ($	ext{rad}$) ↓ |
| :--- | :---: | :---: | :---: |
| **Ground Truth (Human Expert)** | 2.9882 | 0.0812 | 0.0415 |
| **TCP Baseline (Vision-Only)** | 4.8921 | 0.2415 | 0.0982 |
| **CORTEX V2I-Sync (Ideal)** | **3.4543** | **0.1102** | **0.0521** |
| **CORTEX Ego-Only (Standalone)** | 3.4489 | 0.1124 | 0.0538 |
| **CORTEX Delayed ($500	ext{ ms}$ Lag)** | **3.5120** | **0.1179** | **0.0581** |

---

### Table 4: Volumetric V2I Network Traffic & Bandwidth Compression Audit

| Evaluation Corridor | Total Frames | Fused Frames | V2I Activation Rate (%) | Baseline Payload (MB) | CORTEX Payload (MB) | Bandwidth Savings (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Left Turn Complex Corridor** | 311 | 92 | 29.6% | 6297.75 | 4436.15 | **29.6%** |
| **Right Turn Occluded Apex** | 182 | 100 | 54.9% | 3685.50 | 1662.22 | **54.9%** |

---

### Table 5: Pose Uncertainty Sensitivity Analysis ($\mathcal{N}(0, \sigma^2)$)

| Localization Drift ($\sigma$) | Operating Condition | TCP Baseline ADE (m) | CORTEX ADE (m) | Control $L_1$ Dev | Robustness Gain |
| :--- | :--- | :---: | :---: | :---: | :---: |
| $\sigma = 0.0	ext{ m}$ | Ideal Reference State | 0.68 | **0.3157** | 0.0658 | **+53.5%** |
| $\sigma = 0.2	ext{ m}$ | Operational Safety Limit | 0.84 | **0.3179** | 0.0658 | **+62.1%** |
| $\sigma = 0.5	ext{ m}$ | Critical GNSS Degradation | 1.15 | **0.3179** | 0.0658 | **+72.3%** |

---

## 💻 Cross-Platform & Windows Compatibility Notes

To ensure seamless execution across both Windows and Linux environments, the following technical conventions are integrated into all repository scripts:

1. **Path Normalization:** All file I/O operations utilize `pathlib.Path` or `os.path.join` to avoid backslash/slash path delimiter failures on Windows.
2. **CUDA IPC & Dataloader Worker Compatibility:** On Windows platforms, `num_workers` in `torch.utils.data.DataLoader` defaults to `0` or `2` to prevent Windows-specific `spawn` process hanging issues.
3. **Automatic Mixed Precision (AMP):** Utilizes `torch.cuda.amp.autocast()` to optimize VRAM utilization on Windows systems without requiring custom CUDA extensions.

---

## 🔗 Citation

If you find CORTEX useful in your research or autonomous driving engineering workflows, please cite our official paper:

```bibtex
@article{daneshvar2025cortex,
  title   = {CORTEX: Cooperative Occlusion-Resilient Trajectory Execution via Request-Aware V2I Fusion},
  author  = {Daneshvar, Hamid and Masih-Tehrani, Masoud and Mollajafari, Morteza},
  journal = {IEEE Access},
  volume  = {13},
  pages   = {1--18},
  year    = {2025},
  doi     = {10.1109/ACCESS.2024.DoiNumber}
}
```

---

## 📄 License & Acknowledgements

- **License:** Distributed under the **MIT License**. See `LICENSE` for details.
- **Acknowledgements:** This repository builds upon the foundational trajectory-guided concepts established by **TCP** ([NeurIPS 2022](https://github.com/OpenDriveLab/TCP)) and utilizes the collaborative simulation tools of **V2XVerse** ([PAMI 2025](https://github.com/gjliu/V2Xverse)).
