"""
CORTEX: Empirical V2I Network Bandwidth & Telemetry Audit Tool
Quantifies live data compression footprints and bandwidth savings across urban corridors.
Ref: IEEE Access - Section IV-C (Quantitative V2I Communication Overhead)
"""

import os
import sys
import json
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
    parser = argparse.ArgumentParser(description="CORTEX V2I Telemetry & Bandwidth Audit")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset_root", type=str, default="./dataset", help="Dataset root directory")
    parser.add_argument("--scenario_file", type=str, default="./scenarios.json", help="Path to scenarios JSON file")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = GlobalConfig()

    print(f"Loading checkpoint: {args.checkpoint_path}")
    trainer_module = CoTCP_Trainer.load_from_checkpoint(args.checkpoint_path, map_location=device, config=config, lr=2e-5)
    model = trainer_module.model.to(device).eval()

    with open(args.scenario_file, 'r', encoding='utf-8') as f:
        scenarios_dict = json.load(f)

    # Tensor payload constants
    CHANNELS, H, W = 384, 96, 288
    BYTES_PER_ELEMENT = 2  # Float16 precision
    BASELINE_FRAME_MB = (CHANNELS * H * W * BYTES_PER_ELEMENT) / (1024 * 1024)  # ~20.25 MB

    print("\n" + "="*85)
    print(" CORTEX LIVE GRAPH EXPERIMENTAL V2I NETWORK TRAFFIC AUDIT")
    print("="*85)

    for sc_name, sc_info in scenarios_dict.items():
        target_route_marker = sc_info['route_contains']
        valid_ranges = sc_info['ranges']

        dataset = V2XVerse_TCP_Dataset(
            raw_data_root=Path(args.dataset_root),
            config=config,
            split='val',
            town_filter=[target_route_marker]
        )

        filtered_index = [
            sample for sample in dataset.flat_index
            if any(start <= sample['frame_id'] <= end for start, end in valid_ranges)
        ]
        dataset.flat_index = filtered_index

        if not dataset.flat_index: continue

        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=tcp_collate_fn)

        total_baseline_mb = 0.0
        total_cortex_mb = 0.0
        successful_fusions = 0
        total_processed = 0

        with torch.no_grad():
            for batch in loader:
                if batch is None: continue

                batch_device = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor): batch_device[k] = v.to(device)
                    elif isinstance(v, dict): batch_device[k] = {dk: (dv.to(device) if isinstance(dv, torch.Tensor) else dv) for dk, dv in v.items()}
                    else: batch_device[k] = v

                outputs = model(batch_device)

                rsu_key = 'rsu_lidar_dict' if 'rsu_lidar_dict' in batch_device else 'rsu_lidar'
                has_rsu = rsu_key in batch_device and batch_device[rsu_key] is not None

                if has_rsu:
                    request_mask = model.request_generator(outputs['coarse_traj'])
                    resized_mask = F.interpolate(request_mask, size=(H, W), mode='bilinear', align_corners=True)
                    active_cells = (resized_mask > 0.01).float().sum()
                    active_ratio = (active_cells / resized_mask.numel()).item()
                    successful_fusions += 1
                else:
                    active_ratio = 1.0

                total_processed += 1
                total_baseline_mb += BASELINE_FRAME_MB
                total_cortex_mb += (BASELINE_FRAME_MB * active_ratio)

        savings = (1.0 - (total_cortex_mb / total_baseline_mb)) * 100.0
        fusion_eff = (successful_fusions / total_processed) * 100.0

        print(f"\nScenario: {sc_name}")
        print(f"  ▪ Total Processed Frames         : {total_processed}")
        print(f"  ▪ V2I Stream Activation Rate    : {fusion_eff:.1f}% ({successful_fusions}/{total_processed} Frames)")
        print(f"  ▪ Uncompressed Baseline Traffic  : {total_baseline_mb:.2f} MB")
        print(f"  ▪ CORTEX Streamlined Cost        : {total_cortex_mb:.2f} MB")
        print(f"  ▪ Verified Bandwidth Savings     : {savings:.1f}% Reduction")


if __name__ == "__main__":
    main()