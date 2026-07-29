"""
CORTEX: Cooperative Occlusion-Resilient Trajectory Execution via Request-Aware V2I Fusion
Main Offline Ablation Matrix Evaluation Engine on CARLA Town05 Benchmark
"""

import os
import sys
import json
import math
import random
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

import numpy as np
import torch
import torch.utils.checkpoint as cp
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from scipy.signal import savgol_filter

# Cross-platform environment and module setup
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
cp.checkpoint = lambda fn, *args, **kwargs: fn(*args, **kwargs)

from config import GlobalConfig
from data import V2XVerse_TCP_Dataset, tcp_collate_fn
from train import CoTCP_Trainer

DT = 0.2


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def sanitize_for_json(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_json(v) for v in data]
    elif isinstance(data, (float, np.float32, np.float64)):
        if np.isnan(data): return 'NaN'
        elif np.isposinf(data): return 'Infinity'
        elif np.isneginf(data): return '-Infinity'
        return float(data)
    elif isinstance(data, (int, np.int32, np.int64)):
        return int(data)
    elif isinstance(data, np.ndarray):
        return sanitize_for_json(data.tolist())
    elif torch.is_tensor(data):
        return sanitize_for_json(data.detach().cpu().numpy())
    return data


def to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor): out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = {nk: (nv.to(device) if isinstance(nv, torch.Tensor) else nv) for nk, nv in v.items()}
        else: out[k] = v
    return out


def indices_for_scenario(dataset: V2XVerse_TCP_Dataset, scen: Dict) -> List[int]:
    route_substr = scen["route_contains"]
    out = []
    if not hasattr(dataset, 'flat_index') or not dataset.flat_index: return []
    for i, info in enumerate(dataset.flat_index):
        if route_substr in info.get("route_path_str", ""):
            try:
                fid = int(info["frame_id"])
                for a, b in scen["ranges"]:
                    if a <= fid <= b:
                        out.append(i)
                        break
            except ValueError:
                continue
    return out


def group_scenario_indices(dataset: V2XVerse_TCP_Dataset, scenario_idxs: List[int]) -> Dict[str, List[int]]:
    runs = defaultdict(list)
    valid_indices = [i for i in scenario_idxs if isinstance(dataset.flat_index[i], dict)]
    sorted_idxs = sorted(valid_indices, key=lambda idx: (dataset.flat_index[idx]['route_path_str'], int(dataset.flat_index[idx]['frame_id'])))

    for i in sorted_idxs:
        runs[dataset.flat_index[i]['route_path_str']].append(i)

    contiguous_runs = {}
    run_counter = defaultdict(int)
    for r_key, indices in runs.items():
        if not indices: continue
        c_run_key = f"{r_key}_run_{run_counter[r_key]}"
        contiguous_runs[c_run_key] = [indices[0]]
        for k in range(1, len(indices)):
            prev, curr = indices[k-1], indices[k]
            is_consec = int(dataset.flat_index[curr]['frame_id']) <= int(dataset.flat_index[prev]['frame_id']) + 2
            if is_consec:
                contiguous_runs[c_run_key].append(curr)
            else:
                run_counter[r_key] += 1
                c_run_key = f"{r_key}_run_{run_counter[r_key]}"
                contiguous_runs[c_run_key] = [curr]
    return {k: v for k, v in contiguous_runs.items() if len(v) >= 10}


def evaluate_ablation_matrix_run(run_indices, dataset, model, device, scenario_name, run_id, mode, latency_ms):
    results_list = []
    run_subset = Subset(dataset, run_indices)
    run_loader = DataLoader(run_subset, batch_size=1, shuffle=False, num_workers=0, collate_fn=tcp_collate_fn)
    model.eval()

    desc_str = f"      -> [{mode:<15} | Delay: {latency_ms:3d}ms]"

    with torch.no_grad():
        for i, batch in enumerate(tqdm(run_loader, desc=desc_str, leave=False)):
            current_original_index = run_indices[i]
            base_ds = dataset
            while isinstance(base_ds, Subset):
                base_ds = base_ds.dataset
            info = base_ds.flat_index[current_original_index]
            frame_id = info['frame_id']

            frame_data = {
                "scenario": scenario_name, "run_id": run_id, "frame_id": frame_id,
                "timestamp": i * DT, "evaluation_mode": mode, "injected_latency_ms": latency_ms,
                "ground_truth": {}, "prediction": {}
            }

            if batch is None:
                frame_data["status"] = "skipped_none_batch"
                results_list.append(frame_data)
                continue

            try:
                gt_wp_t = batch.get("waypoints_gt")
                frame_data["ground_truth"]["gt_path_xy"] = gt_wp_t.squeeze(0).cpu().numpy().tolist() if gt_wp_t is not None else []
                frame_data["ground_truth"]["gt_ctrl"] = batch.get("control_gt").squeeze(0).cpu().numpy().tolist() if batch.get("control_gt") is not None else []
            except Exception as e:
                frame_data["status"] = f"error_gt: {e}"
                results_list.append(frame_data)
                continue

            if mode == "Ego-only":
                batch['rsu_lidar_dict'] = None

            latency_sec = latency_ms / 1000.0
            if mode == "CORTEX-Ultimate":
                batch['communication_delay'] = torch.tensor([latency_sec], dtype=torch.float32)
            elif mode == "V2I-Fixed-Sync":
                batch['communication_delay'] = torch.tensor([0.0], dtype=torch.float32)

            try:
                cuda_batch = to_device(batch, device)
                outputs = model(cuda_batch)
                pred_wp_np = outputs["pred_wp"].squeeze(0).detach().cpu().numpy()
                pred_ctrl_np = outputs["pred_ctrl"].squeeze(0).detach().cpu().numpy()

                frame_data["prediction"] = {
                    "pred_path_xy": pred_wp_np.tolist(),
                    "pred_ctrl_throttle": float(pred_ctrl_np[0]),
                    "pred_steer": float(pred_ctrl_np[1]),
                    "pred_ctrl_brake": float(pred_ctrl_np[2])
                }
            except Exception as e:
                frame_data["prediction"]["status"] = f"error_model_run: {e}"

            results_list.append(frame_data)

    return results_list


def main():
    parser = argparse.ArgumentParser(description="CORTEX Matrix Offline Evaluation Engine")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to trained .ckpt file")
    parser.add_argument("--dataset_root", type=str, default="./dataset", help="Root directory of dataset")
    parser.add_argument("--scenario_file", type=str, default="./scenarios.json", help="Path to scenarios JSON file")
    parser.add_argument("--towns", nargs="+", default=["town05"], help="Evaluation towns")
    parser.add_argument("--output_file", type=str, default="cortex_ultimate_ablation_matrix_results.json", help="Output JSON path")
    args = parser.parse_args()

    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = GlobalConfig()

    print(f"Loading checkpoint: {args.checkpoint_path}")
    trainer_module = CoTCP_Trainer.load_from_checkpoint(args.checkpoint_path, map_location=device, config=cfg, lr=2e-5)
    model = trainer_module.model.to(device).eval()

    ds = V2XVerse_TCP_Dataset(raw_data_root=Path(args.dataset_root), config=cfg, split="val", town_filter=args.towns)

    with open(args.scenario_file, "r", encoding="utf-8") as f:
        all_scenarios = json.load(f)

    global_ablation_results = []
    latency_steps = [0, 100, 200, 300, 400, 500]
    ablation_modes = ["Ego-only", "V2I-Fixed-Sync", "CORTEX-Ultimate"]

    print("\n" + "="*80 + "\n INITIATING CORTEX ABLATION MATRIX EVALUATION \n" + "="*80)

    for name, details in all_scenarios.items():
        print(f"\n>>> TARGET SCENARIO: {name}")
        scen_indices = indices_for_scenario(ds, details)
        if not scen_indices: continue

        scenario_runs_map = group_scenario_indices(ds, scen_indices)
        for run_id, run_idxs in scenario_runs_map.items():
            for mode in ablation_modes:
                delays = latency_steps if mode == "CORTEX-Ultimate" else [0]
                for delay in delays:
                    frames = evaluate_ablation_matrix_run(run_idxs, ds, model, device, name, run_id, mode, delay)
                    if frames:
                        global_ablation_results.extend(frames)

    clean_data = sanitize_for_json(global_ablation_results)
    with open(args.output_file, "w", encoding='utf-8') as f:
        json.dump(clean_data, f, indent=2)
    print(f"\n✅ Matrix results successfully saved to: {args.output_file}")


if __name__ == "__main__":
    main()