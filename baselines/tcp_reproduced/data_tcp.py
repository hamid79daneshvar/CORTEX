# baselines/tcp_reproduced/data_tcp.py
# Official Data Loader for TCP Baseline on V2XVerse Dataset
# Adapts V2XVerse data structure to match TCP's expected input format.

import os
import json
import numpy as np
import math
import traceback
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

class V2XVerse_for_TCP_Dataset(Dataset):
    """
    Dataset loader for the TCP Baseline.
    Loads front camera images and navigational measurements from V2XVerse.
    """
    def __init__(self, root, config, split='val', town_filter=None):
        self.root = Path(root)
        self.config = config
        self.is_train = split == 'train'
        self.pred_len = config.pred_len
        
        # Image Transformations (Standard ImageNet Normalization)
        self.transform = T.Compose([
            T.Resize((config.input_resolution, config.input_resolution)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load Dataset Index
        index_file_path = self.root / 'dataset_index.txt'
        if not index_file_path.exists():
            raise FileNotFoundError(f"Index file not found at {index_file_path}")

        with open(index_file_path, 'r') as f:
            all_samples_info = f.readlines()
        
        # Filter by towns
        if town_filter:
            self.routes = [line for line in all_samples_info if any(town in line for town in town_filter)]
        else:
            self.routes = all_samples_info
        
        # Build Flat Index
        self.flat_index = []
        for line in tqdm(self.routes, desc=f"Indexing TCP routes for split '{split}'"):
            parts = line.strip().split()
            if len(parts) < 2: continue
            
            route_path_str = parts[0]
            seq_len = int(parts[1])
            
            # Skip first few frames to ensure history, and last few for prediction horizon
            start_frame = 5
            end_frame = seq_len - self.pred_len - 5
            
            for frame_id in range(start_frame, end_frame):
                self.flat_index.append({
                    'path': route_path_str,
                    'frame_id': frame_id
                })
        
        print(f"[{split.upper()}] Loaded {len(self.flat_index)} frames.")

    def __len__(self):
        return len(self.flat_index)

    def __getitem__(self, idx):
        try:
            sample_info = self.flat_index[idx]
            seq_path = self.root / sample_info['path']
            frame_id = sample_info['frame_id']
            ego_path = seq_path / 'ego_vehicle_0'

            # -------------------------------------
            # 1. Load Images
            # -------------------------------------
            # TCP uses only the front RGB camera
            img_path = ego_path / 'rgb_front' / f'{frame_id:04d}.png'
            if not img_path.exists():
                # Fallback to jpg if png not found
                img_path = ego_path / 'rgb_front' / f'{frame_id:04d}.jpg'
            
            front_img = Image.open(img_path).convert('RGB')
            front_img_tensor = self.transform(front_img)

            # -------------------------------------
            # 2. Load Measurements & Ego State
            # -------------------------------------
            with open(ego_path / 'measurements' / f'{frame_id:04d}.json') as f:
                meas_data = json.load(f)
            
            ego_pose = np.array([meas_data['x'], meas_data['y']])
            yaw_rad = math.radians(meas_data['yaw'])
            
            speed = torch.tensor([meas_data['speed']], dtype=torch.float32)
            
            # Command (One-Hot Encoding or Index)
            # TCP usually expects a one-hot vector or integer. 
            # Here we simplify to a float embedding input if the model supports it,
            # or map to standard CARLA commands: 
            # 1=Left, 2=Right, 3=Straight, 4=Follow, 5=ChangeLeft, 6=ChangeRight
            cmd_val = meas_data.get('command', 2.0)
            cmd_vec = torch.zeros(6)
            cmd_idx = int(cmd_val) - 1
            if 0 <= cmd_idx < 6:
                cmd_vec[cmd_idx] = 1.0
            else:
                cmd_vec[2] = 1.0 # Default to Straight/Follow

            # -------------------------------------
            # 3. Future Waypoints (Ground Truth)
            # -------------------------------------
            waypoints_gt = []
            for i in range(1, self.pred_len + 1):
                # Load future measurement for accurate position
                future_file = ego_path / 'measurements' / f'{frame_id + i:04d}.json'
                if os.path.exists(future_file):
                    with open(future_file) as f:
                        future_meas = json.load(f)
                    future_pose = np.array([future_meas['x'], future_meas['y']])
                else:
                    future_pose = ego_pose # Fallback
                
                # Global to Local Coordinate Transformation
                # (Same logic as in CORTEX data loader)
                delta_future = future_pose - ego_pose
                wx = delta_future[0] * math.cos(-yaw_rad) - delta_future[1] * math.sin(-yaw_rad)
                wy = delta_future[0] * math.sin(-yaw_rad) + delta_future[1] * math.cos(-yaw_rad)
                waypoints_gt.append([wx, wy])
            
            # Target point is the last waypoint
            target_point = torch.tensor(waypoints_gt[-1], dtype=torch.float32)
            
            # -------------------------------------
            # 4. Control Ground Truth
            # -------------------------------------
            control_gt = torch.tensor([
                meas_data['throttle'], 
                meas_data['steer'], 
                meas_data['brake']
            ], dtype=torch.float32)

            return {
                'front_img': front_img_tensor,
                'speed': speed,
                'target_point': target_point,
                'target_command': cmd_vec,
                'waypoints': torch.tensor(waypoints_gt, dtype=torch.float32),
                'control_gt': control_gt
            }

        except Exception as e:
            # print(f"Error loading baseline frame {idx}: {e}")
            return None

def tcp_collate_fn(batch):
    """
    Collate function to filter None samples and stack tensors.
    """
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0: return None
    
    return {
        'front_img': torch.stack([b['front_img'] for b in batch]),
        'speed': torch.stack([b['speed'] for b in batch]),
        'target_point': torch.stack([b['target_point'] for b in batch]),
        'target_command': torch.stack([b['target_command'] for b in batch]),
        'waypoints': torch.stack([b['waypoints'] for b in batch]),
        'control_gt': torch.stack([b['control_gt'] for b in batch])
    }