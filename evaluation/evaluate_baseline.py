# evaluation/evaluate_baseline.py
# Official Evaluation Script for TCP Baseline (Vision-Only)
# Standardized to match CORTEX evaluation protocol.

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

# --- 1. Path Setup (CRITICAL) ---
# We need to access the baseline code located in 'baselines/tcp_reproduced'
# Structure: project_root/evaluation/evaluate_baseline.py
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
baseline_dir = os.path.join(project_root, 'baselines', 'tcp_reproduced')

sys.path.append(project_root) # For general access
sys.path.append(baseline_dir) # To import 'model', 'data_tcp', 'config' from baseline folder

# Setup Environment
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Import Baseline Modules (Dynamically loaded from baseline_dir)
try:
    from model import TCP
    from data_tcp import V2XVerse_for_TCP_Dataset, tcp_collate_fn
    from config import GlobalConfig as BaselineConfig
except ImportError as e:
    print(f"Error importing baseline modules: {e}")
    print(f"Ensure that 'model.py', 'data_tcp.py', and 'config.py' exist in {baseline_dir}")
    sys.exit(1)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_baseline_model(checkpoint_path, device):
    """
    Loads the TCP model from a Lightning checkpoint or raw state dict.
    """
    config = BaselineConfig()
    model = TCP(config)
    
    print(f"Loading weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle Lightning Checkpoint structure
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Remove 'model.' prefix if saved via LightningModule wrapper
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    else:
        # Assume raw state dict
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    return model, config

def evaluate(args):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Baseline evaluation on {device}...")

    # 1. Load Model
    model, config = load_baseline_model(args.checkpoint_path, device)

    # 2. Load Scenarios & Ranges from JSON
    # Structure: { "scenario_name": { "route_contains": "...", "ranges": [[start, end], ...] } }
    target_routes_info = {} 
    
    if args.scenario_file and os.path.exists(args.scenario_file):
        with open(args.scenario_file, 'r') as f:
            scenarios_data = json.load(f)
        
        print(f"Loaded {len(scenarios_data)} scenarios from JSON.")
        
        for sc_name, sc_info in scenarios_data.items():
            route_key = sc_info['route_contains']
            ranges = sc_info.get('ranges', []) 
            target_routes_info[route_key] = ranges
    else:
        print("Warning: No scenario file provided. Evaluating ALL frames (Full Routes).")

    # 3. Load Dataset (TCP Version)
    print(f"Loading V2XVerse Dataset for TCP (Towns: {args.towns})...")
    try:
        test_set = V2XVerse_for_TCP_Dataset(
            root=args.data_root, 
            config=config, 
            split='val', 
            town_filter=args.towns
        )
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return

    # 4. Filter Dataset based on Scenarios AND Ranges
    valid_indices = []
    
    if target_routes_info:
        print("Filtering dataset based on scenarios and frame ranges...")
        # Access flat_index directly from V2XVerse_for_TCP_Dataset
        for i, sample in enumerate(test_set.flat_index):
            # sample['path'] is route path, sample['frame_id'] is int
            
            matched_key = None
            for route_key in target_routes_info.keys():
                if route_key in sample['path']:
                    matched_key = route_key
                    break
            
            if matched_key:
                frame_id = sample['frame_id']
                ranges = target_routes_info[matched_key]
                is_in_range = False
                
                if not ranges: # If no ranges, take full route
                    is_in_range = True
                else:
                    for r in ranges:
                        if r[0] <= frame_id <= r[1]:
                            is_in_range = True
                            break
                
                if is_in_range:
                    valid_indices.append(i)
        
        print(f"Filtered {len(valid_indices)} frames matching the scenarios.")
        if len(valid_indices) == 0:
            print("Error: No frames matched. Check scenario file vs dataset.")
            return
        
        subset = Subset(test_set, valid_indices)
        # IMPORTANT: shuffle=False to maintain sequence order for plotting
        dataloader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, 
                                num_workers=args.num_workers, collate_fn=tcp_collate_fn)
    else:
        dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, 
                                num_workers=args.num_workers, collate_fn=tcp_collate_fn)

    # 5. Inference Loop
    results_list = []
    print("Starting Baseline Inference...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if batch is None: continue
            
            # Move data to GPU
            img = batch['front_img'].to(device)
            speed = batch['speed'].to(device)
            target_point = batch['target_point'].to(device)
            command = batch['target_command'].to(device)
            control_gt = batch['control_gt'].to(device) # [Throttle, Steer, Brake]
            
            # Prepare measurements input for TCP
            measurements = torch.cat([speed, target_point, command], dim=1)
            
            # Forward
            pred = model(img, measurements, target_point)
            
            batch_size = img.shape[0]
            for i in range(batch_size):
                # TCP Output: [Throttle, Steer, Brake]
                pred_ctrl = pred['pred_ctrl'][i].cpu().numpy()
                gt_ctrl = control_gt[i].cpu().numpy()
                
                # Get current speed (Essential for kinematic calculations later)
                current_speed = float(speed[i].item())

                frame_data = {
                    "pred_steer": float(pred_ctrl[1]),
                    "gt_steer": float(gt_ctrl[1]),
                    "pred_throttle": float(pred_ctrl[0]),
                    "gt_throttle": float(gt_ctrl[0]),
                    "pred_brake": float(pred_ctrl[2]),
                    "gt_brake": float(gt_ctrl[2]),
                    "speed": current_speed # Added for Lat Accel calc
                }
                results_list.append(frame_data)

    # 6. Save Results
    with open(args.output_file, 'w') as f:
        json.dump(results_list, f, indent=4)
    
    print(f"Baseline evaluation complete. Results saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TCP Baseline")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to TCP .ckpt or .pth file")
    parser.add_argument("--data_root", type=str, default="./dataset", help="Path to V2XVerse dataset")
    parser.add_argument("--scenario_file", type=str, default="scenarios.json", help="JSON file with scenario definitions")
    parser.add_argument("--towns", nargs="+", default=["town05"], help="Towns to evaluate")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_file", type=str, default="tcp_eval_results.json")
    
    args = parser.parse_args()
    
    evaluate(args)