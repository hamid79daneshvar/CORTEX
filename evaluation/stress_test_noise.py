"""
CORTEX: Spatial Localization Pose Noise Robustness Stress Test
Evaluates CORTEX trajectory and control resilience under additive Gaussian pose noise.
Ref: IEEE Access - Section IV-E (Generalized Sensitivity Analysis Under Pose Uncertainties)
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

from config import GlobalConfig
from data import V2XVerse_TCP_Dataset, tcp_collate_fn
from train import CoTCP_Trainer


def main():
    parser = argparse.ArgumentParser(description="CORTEX Localization Noise Sensitivity Stress Test")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset_root", type=str, default="./dataset", help="Dataset root path")
    parser.add_argument("--towns", nargs="+", default=["town05"], help="Evaluation towns")
    parser.add_argument("--noise_stds", nargs="+", type=float, default=[0.0, 0.2, 0.5], help="Gaussian noise standard deviations (m)")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = GlobalConfig()

    print(f"Loading checkpoint: {args.checkpoint_path}")
    trainer_module = CoTCP_Trainer.load_from_checkpoint(args.checkpoint_path, map_location=device, config=config, lr=2e-5)
    model = trainer_module.model.to(device).eval()

    dataset = V2XVerse_TCP_Dataset(raw_data_root=Path(args.dataset_root), config=config, split='val', town_filter=args.towns)
    print(f"Target Town05 dataset loaded with {len(dataset)} frames.")

    results_summary = {}

    print("\n" + "="*85)
    print(" CORTEX SPATIAL LOCALIZATION NOISE STRESS TEST (IEEE Access Standard)")
    print("="*85)

    for noise in args.noise_stds:
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, collate_fn=tcp_collate_fn)

        total_wp_l2 = 0.0
        total_ctrl_l1 = 0.0
        valid_batches = 0

        with torch.no_grad():
            for batch in loader:
                if batch is None: continue

                batch_device = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch_device[k] = v.to(device)
                    elif isinstance(v, dict):
                        batch_device[k] = {dk: (dv.to(device) if isinstance(dv, torch.Tensor) else dv) for dk, dv in v.items()}
                    else:
                        batch_device[k] = v

                # Inject dynamic spatial pose coordinate noise
                if noise > 0.0 and 'rsu_lidar_dict' in batch_device and batch_device['rsu_lidar_dict'] is not None:
                    noise_tensor = torch.randn_like(batch_device['rsu_lidar_dict']['voxel_features'][:, :, :2]) * noise
                    batch_device['rsu_lidar_dict']['voxel_features'][:, :, :2] += noise_tensor

                outputs = model(batch_device)
                diff = outputs['pred_wp'] - batch_device['waypoints_gt']
                l2_distances = torch.norm(diff, p=2, dim=-1)

                wp_err = l2_distances.mean().item()
                ctrl_err = F.l1_loss(outputs['pred_ctrl'], batch_device['control_gt']).item()

                total_wp_l2 += wp_err
                total_ctrl_l1 += ctrl_err
                valid_batches += 1

        avg_ade = total_wp_l2 / valid_batches if valid_batches > 0 else 0.0
        avg_ctrl = total_ctrl_l1 / valid_batches if valid_batches > 0 else 0.0
        results_summary[noise] = {'ADE': avg_ade, 'Control_L1': avg_ctrl}

        print(f"Noise Std {noise:.1f} m -> Mean Trajectory ADE: {avg_ade:.4f} m | Mean Control L1: {avg_ctrl:.4f}")

    print("\n" + "="*85)
    print(f"{'Spatial Noise Level (m)':<25}{'Trajectory ADE (m)':<30}{'Control L1 Deviation'}")
    print("-"*85)
    for n_val, metrics in results_summary.items():
        print(f"{n_val:<25.1f}{metrics['ADE']:<30.4f}{metrics['Control_L1']:.4f}")
    print("="*85)


if __name__ == "__main__":
    main()
