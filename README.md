# CORTEX: Cooperative Occlusion-Resilient Trajectory Execution via Request-Aware V2I Fusion

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/IEEE-Access-blue)](https://ieeeaccess.ieee.org/)

This repository is the official PyTorch implementation of the paper:
**"CORTEX: Cooperative Occlusion-Resilient Trajectory Execution via Request-Aware V2I Fusion"**
*Submitted to IEEE Access.*

## 📝 Abstract

Although end-to-end (E2E) autonomous driving successfully maps sensor inputs to control actions in clear conditions, it hits a hard physical wall in dense cities: occlusion. When facing blind intersections, single-vehicle agents suffer from high uncertainty, often causing them to freeze or drive erratically.

To fix this perception gap, we propose **CORTEX**. This framework extends the standard Trajectory-guided Control Prediction (TCP) architecture by integrating Vehicle-to-Infrastructure (V2I) data. Unlike systems that broadcast everything indiscriminately, CORTEX uses a **Request-Aware Spatial Fusion** mechanism to query Roadside Units (RSUs) only for relevant spatial features.

Experiments on the V2XVerse benchmark (Town05) show that CORTEX reduces control signal noise (RMS Yaw Rate) by **74.1%** and improves heading accuracy in blind turns by **43.0%**.

## 🎥 Qualitative Results (Demos)

Performance of CORTEX in challenging occluded scenarios (Town05 Test Set).

| **Scenario 1: Unprotected Left Turn** | **Scenario 2: Blind Right Turn** |
| :---: | :---: |
| ![Left Turn](assets/demo_left.gif) | ![Right Turn](assets/demo_right.gif) |
| *Smooth navigation with dynamic traffic* | *Stable control despite occlusion* |

## 🏗️ Architecture

The CORTEX framework consists of a dual-branch control architecture enhanced by Request-Aware V2I collaboration.

![Architecture](assets/architecture.png)

## 📂 Project Structure

```text
CORTEX_Repo/
├── cortex/               # Source code for the proposed CORTEX model (Ours)
├── baselines/            # Re-implementation of the TCP Baseline (Vision-Only)
├── evaluation/           # Scripts for kinematic metrics and plotting
├── dataset/              # Dataset directory (see Data Preparation)
├── gen_index.py          # Data preprocessing script
├── requirements.txt      # Python dependencies
└── README.md             # Documentation

## 🛠️ Installation
We recommend using Anaconda to manage the environment.

System Requirements:

OS: Linux or Windows 10/11

Python: 3.7 (Recommended)

GPU: NVIDIA RTX 3090 (or equivalent with 24GB VRAM)

CUDA: 11.3

1. Clone the repository
git clone [https://github.com/hamid79daneshvar/CORTEX.git](https://github.com/hamid79daneshvar/CORTEX.git)
cd CORTEX

2. Create Environment
conda create -n cortex python=3.7 -y
conda activate cortex

3. Install Dependencies
First, install PyTorch with CUDA 11.3 support:
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url [https://download.pytorch.org/whl/cu113](https://download.pytorch.org/whl/cu113)
