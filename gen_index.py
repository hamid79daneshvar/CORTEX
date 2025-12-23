# gen_index.py
# Official Data Indexing Script for CORTEX
# Scans the V2XVerse dataset directory and generates 'dataset_index.txt'.

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Generate dataset index for V2XVerse")
    parser.add_argument('--root', type=str, default='dataset', help='Root directory containing weather-0, etc.')
    parser.add_argument('--output', type=str, default='dataset_index.txt', help='Output index filename')
    parser.add_argument('--min_len', type=int, default=50, help='Minimum sequence length to include')
    # Safety buffer: reduces the indexed length by this amount to avoid end-of-sequence issues
    parser.add_argument('--buffer', type=int, default=25, help='Frame buffer to subtract from total length')
    return parser.parse_args()

def generate_index(args):
    dataset_root = Path(args.root)
    output_path = dataset_root / args.output
    
    if not dataset_root.exists():
        print(f"Error: Dataset root '{dataset_root}' does not exist.")
        print("Please ensure your folder structure is: CORTEX_Repo/dataset/weather-0/data/...")
        sys.exit(1)

    # Dictionary to store the best sequence for each route
    # Key: town_route_id (e.g., 'town05_29'), Value: {path, length, num_agents}
    route_registry = {}
    town_counter = defaultdict(int)
    
    # We assume standard structure: root/weather-X/data/route_folder
    # Find all weather folders
    weather_folders = list(dataset_root.glob('weather-*'))
    if not weather_folders:
        print(f"Warning: No 'weather-*' folders found in {dataset_root}.")
        # Fallback: maybe the user pointed directly to 'data'? 
        # For safety, we search recursively for 'data' folders.
        data_roots = list(dataset_root.rglob('data'))
    else:
        data_roots = [w / 'data' for w in weather_folders]

    print(f"Scanning {len(data_roots)} data directories...")
    
    total_scanned = 0
    
    for data_dir in data_roots:
        if not data_dir.exists(): continue
        
        # List all route subdirectories (e.g., routes_town05_29_w0...)
        sub_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
        
        for sub_dir in tqdm(sub_dirs, desc=f"Scanning {data_dir.name}"):
            try:
                agent_list = [p.name for p in sub_dir.iterdir() if p.is_dir()]
            except Exception:
                continue

            ego_list = [a for a in agent_list if a.startswith('ego')]
            rsu_list = [a for a in agent_list if a.startswith('rsu')]
            
            # Skip if no valid agents
            if not ego_list: # Baseline needs at least one ego
                continue
            
            # Calculate sequence length based on the first ego vehicle's camera frames
            # (Assuming all agents have sync frames, taking the min length is safer)
            seq_len = float('inf')
            
            valid_seq = False
            for ego in ego_list:
                rgb_path = sub_dir / ego / 'rgb_front'
                if rgb_path.exists():
                    current_len = len(list(rgb_path.glob('*.png'))) + len(list(rgb_path.glob('*.jpg')))
                    if current_len < seq_len:
                        seq_len = current_len
                    valid_seq = True
            
            if not valid_seq or seq_len == float('inf'):
                continue

            # Filter short sequences
            if seq_len > args.min_len:
                # Extract ID: routes_town05_29_... -> town05_29
                parts = sub_dir.name.split('_')
                if len(parts) > 2:
                    town_route_id = f"{parts[1]}_{parts[2]}"
                    town_name = parts[1] # e.g. town05
                else:
                    # Fallback naming if structure is different
                    town_route_id = sub_dir.name
                    town_name = "unknown"

                # Logic: If we have duplicates (same route recorded twice), keep the longer one
                if town_route_id not in route_registry:
                    route_registry[town_route_id] = {
                        'path': sub_dir.relative_to(dataset_root).as_posix(), # Store relative path (Linux style)
                        'seq_len': seq_len,
                        'agents': len(ego_list)
                    }
                    town_counter[town_name] += 1
                else:
                    if seq_len > route_registry[town_route_id]['seq_len']:
                        route_registry[town_route_id] = {
                            'path': sub_dir.relative_to(dataset_root).as_posix(),
                            'seq_len': seq_len,
                            'agents': len(ego_list)
                        }
            
            total_scanned += 1

    # --- Write to Index File ---
    print(f"\nWriting index to {output_path}...")
    with open(output_path, 'w') as f:
        for route_id, info in route_registry.items():
            # Apply safety buffer (from original logic: reduce length by 25)
            final_len = max(0, info['seq_len'] - args.buffer)
            
            if final_len > 0:
                # Format: relative_path sequence_length num_agents
                f.write(f"{info['path']} {final_len} {info['agents']}\n")

    # --- Summary ---
    print("="*40)
    print("Dataset Index Generation Complete")
    print("="*40)
    print(f"Total Folders Scanned: {total_scanned}")
    print(f"Unique Routes Indexed: {len(route_registry)}")
    print("Breakdown by Town:")
    for town, count in sorted(town_counter.items()):
        print(f"  - {town}: {count} sequences")
    print("="*40)

if __name__ == "__main__":
    args = parse_args()
    generate_index(args)