# evaluation/evaluate_cortex.py
# Official Evaluation Script for CORTEX Model
# Includes: Path correction for imports & Precise scenario filtering based on JSON ranges.

import os
import sys
import argparse
import json
import math
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from scipy.signal import savgol_filter
from collections import defaultdict

# --- 1. Path Correction (CRITICAL) ---
# Add the parent directory (project root) to sys.path to allow importing 'config', 'data', 'train'
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Setup Environment
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Now we can safely import project modules
from config import GlobalConfig
from data import V2XVerse_TCP_Dataset, tcp_collate_fn 
from train import CoTCP_Trainer 

# --- Global Constants ---
DT = 0.2 

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate(args):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running evaluation on {device}...")

    # 1. Load Configuration & Model
    print(f"Loading checkpoint from: {args.checkpoint_path}")
    pl_model = CoTCP_Trainer.load_from_checkpoint(args.checkpoint_path, map_location=device)
    pl_model.eval()
    pl_model.freeze()
    model = pl_model.model.to(device)
    config = pl_model.config_obj

    # 2. Load Scenarios & ranges
    # Structure: { "scenario_name": { "route_contains": "...", "ranges": [[start, end], ...] } }
    target_routes_info = {} # Map route_keyword -> ranges
    
    if args.scenario_file and os.path.exists(args.scenario_file):
        with open(args.scenario_file, 'r') as f:
            scenarios_data = json.load(f)
        
        print(f"Loaded {len(scenarios_data)} scenarios from JSON.")
        
        for sc_name, sc_info in scenarios_data.items():
            route_key = sc_info['route_contains']
            ranges = sc_info.get('ranges', []) 
            # Store ranges. If empty, assume full route (or handle as error depending on logic)
            target_routes_info[route_key] = ranges
    else:
        print("Warning: No scenario file provided. Evaluating ALL frames (Full Routes).")

    # 3. Load Dataset
    print(f"Loading V2XVerse dataset (Towns: {args.towns})...")
    test_set = V2XVerse_TCP_Dataset(
        raw_data_root=args.data_root, 
        config=config, 
        split='val', 
        town_filter=args.towns
    )
    
    # 4. Filter Dataset based on Scenarios AND Ranges
    valid_indices = []
    
    if target_routes_info:
        print("Filtering dataset based on scenarios and frame ranges...")
        for i, sample in enumerate(test_set.indexed_samples):
            # sample['path'] looks like: "weather-0/data/routes_town05_.../..."
            # sample['frame_id'] is the integer frame number
            
            # Check if this sample belongs to any target route
            matched_key = None
            for route_key in target_routes_info.keys():
                if route_key in sample['path']:
                    matched_key = route_key
                    break
            
            if matched_key:
                frame_id = sample['frame_id']
                ranges = target_routes_info[matched_key]
                
                # Check if frame is within any of the defined ranges
                is_in_range = False
                if not ranges: # If no ranges specified, take whole route
                    is_in_range = True
                else:
                    for r in ranges:
                        if r[0] <= frame_id <= r[1]:
                            is_in_range = True
                            break
                
                if is_in_range:
                    valid_indices.append(i)
        
        print(f"Filtered {len(valid_indices)} frames matching the requested scenarios.")
        if len(valid_indices) == 0:
            print("Error: No frames matched. Check scenario file vs dataset paths.")
            return
        
        subset = Subset(test_set, valid_indices)
        # IMPORTANT: shuffle=False to keep sequence order if possible (though Subset might break contiguity)
        dataloader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, 
                                num_workers=args.num_workers, collate_fn=tcp_collate_fn)
    else:
        dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, 
                                num_workers=args.num_workers, collate_fn=tcp_collate_fn)

    # 5. Inference Loop & Logging
    results_list = []
    
    print("Starting Inference...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if batch is None: continue
            
            # Move to device
            batch_gpu = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_gpu[k] = v.to(device)
                elif isinstance(v, dict):
                    batch_gpu[k] = {sk: sv.to(device) if isinstance(sv, torch.Tensor) else sv for sk, sv in v.items()}
                else:
                    batch_gpu[k] = v
            
            # Forward
            pred = model(batch_gpu)
            
            batch_size = batch_gpu['measurements'].shape[0]
            
            for i in range(batch_size):
                # Extract Data
                # Control: [Throttle, Steer, Brake]
                pred_ctrl = pred['pred_ctrl'][i].cpu().numpy()
                gt_ctrl = batch_gpu['control_gt'][i].cpu().numpy()
                
                # Basic Logging (For plotting)
                frame_data = {
                    "pred_steer": float(pred_ctrl[1]),
                    "gt_steer": float(gt_ctrl[1]),
                    "pred_throttle": float(pred_ctrl[0]),
                    "gt_throttle": float(gt_ctrl[0]),
                    "pred_brake": float(pred_ctrl[2]),
                    "gt_brake": float(gt_ctrl[2]),
                    # Note: Without original path/frame_id passed through collate, 
                    # we can't perfectly map back to scenario name here directly.
                    # Assuming sequential processing for plotting.
                }
                results_list.append(frame_data)

    # 6. Save Results
    with open(args.output_file, 'w') as f:
        json.dump(results_list, f, indent=4)
    
    print(f"Evaluation complete. Results saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CORTEX Model")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument("--data_root", type=str, default="./dataset", help="Path to V2XVerse dataset")
    parser.add_argument("--scenario_file", type=str, default="scenarios.json", help="JSON file with scenario definitions")
    parser.add_argument("--towns", nargs="+", default=["town05"], help="Towns to evaluate")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_file", type=str, default="cortex_eval_results.json")
    
    args = parser.parse_args()
    
    evaluate(args)