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


🛠️ InstallationWe recommend using Anaconda to manage the environment.System Requirements:OS: Linux or Windows 10/11Python: 3.7 (Recommended)GPU: NVIDIA RTX 3090 (or equivalent with 24GB VRAM)CUDA: 11.31. Clone the repositoryBashgit clone https://github.com/hamid79daneshvar/CORTEX.git
cd CORTEX
2. Create EnvironmentBashconda create -n cortex python=3.7 -y
conda activate cortex
3. Install DependenciesFirst, install PyTorch with CUDA 11.3 support:Bashpip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
Then, install the remaining requirements:Bashpip install -r requirements.txt
💾 Data PreparationThe model is trained and evaluated on the V2XVerse dataset.Download: Download the dataset from the Official Hugging Face Repository.Organize: Place the data inside a dataset folder in the project root. The structure should look exactly like this:PlaintextCORTEX/dataset/
    └── weather-0/
        └── data/
            ├── routes_town01_...
            ├── ...
            └── routes_town05_...  (Test Set)
Index: Run the indexing script to parse the dataset structure:Bashpython gen_index.py --root dataset
🚀 Training1. Train CORTEX (Ours)To train the cooperative model using LiDAR and V2I fusion:Bashpython cortex/train.py --id cortex_run1 --batch_size 16 --gpus 1
Training Towns: 01, 02, 03, 04, 06.Validation Towns: 07, 10.Test Town: 05 (Reserved for evaluation).2. Train TCP Baseline (Reproduced)To train the vision-only baseline on the exact same split for fair comparison:Bashpython baselines/tcp_reproduced/train_baseline.py --id tcp_baseline --batch_size 32
📈 Evaluation & ReproductionWe provide a complete pipeline to reproduce the quantitative results (Table II) and plots (Fig. 7) presented in the paper.Step 1: Evaluate CORTEXGenerates kinematic predictions for the cooperative model on the test set (Town05).Bashpython evaluation/evaluate_cortex.py \
    --checkpoint_path logs/cortex_run1/checkpoints/best_model.ckpt \
    --output_file results_cortex.json \
    --towns town05
Step 2: Evaluate BaselineGenerates predictions for the vision-only baseline.Bashpython evaluation/evaluate_baseline.py \
    --checkpoint_path logs_baseline/tcp_baseline/checkpoints/best_model.ckpt \
    --output_file results_tcp.json \
    --towns town05
Step 3: Generate Plots and TablesThis script calculates physical metrics (RMS Yaw Rate, Lateral Acceleration) and generates comparison plots.Bashpython evaluation/plot_results.py \
    --cortex_json results_cortex.json \
    --baseline_json results_tcp.json \
    --output_dir ./final_plots
Main Results (Table II)MetricBaseline (TCP)CORTEX (Ours)ImprovementRMS Yaw Rate (°/s)49.6612.83+74.1%Heading Error (°)61.4335.00+43.0%(Results obtained on NVIDIA RTX 3090)🔗 CitationIf you find this code or research useful, please cite our paper:تکه‌کد@article{daneshvar2025cortex,
  title={CORTEX: Cooperative Occlusion-Resilient Trajectory Execution via Request-Aware V2I Fusion},
  author={Daneshvar, Hamid and Masih-Tehrani, Masoud and Mollajafari, Morteza},
  journal={Submitted to IEEE Access},
  year={2025}
}
📄 LicenseThis project is licensed under the MIT License.🙏 AcknowledgementsThis project builds upon the excellent work of TCP (NeurIPS 2022). We thank the authors for releasing their code.
