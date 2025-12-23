# data.py
# Official Data Loading & Preprocessing Pipeline for CORTEX
# Handles V2XVerse dataset, LiDAR voxelization, and coordinate transformations.

import os
import json
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
import math
import random
import traceback

# ==================================================================
# Helper Functions
# ==================================================================

def mask_points_by_range(points, limit_range):
    """
    Filter LiDAR points based on a defined spatial range.
    Args:
        points (np.ndarray): Input point cloud (N, 3+).
        limit_range (list): [x_min, y_min, z_min, x_max, y_max, z_max].
    Returns:
        np.ndarray: Filtered points.
    """
    mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
           & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4]) \
           & (points[:, 2] >= limit_range[2]) & (points[:, 2] <= limit_range[5])
    return points[mask]

def mask_ego_points(points):
    """
    Remove points belonging to the ego-vehicle itself (self-occlusion mask).
    Assuming vehicle box approximately [-1.5, 1.5] x [-1.0, 1.0].
    """
    mask = (points[:, 0] < -1.5) | (points[:, 0] > 1.5) | \
           (points[:, 1] < -1.0) | (points[:, 1] > 1.0)
    return points[mask]

def x1_to_x2(x1_pose, x2_pose):
    """
    Calculate the 4x4 transformation matrix to transform points from 
    coordinate system x1 to x2.
    
    Args:
        x1_pose (list): Pose of source [x, y, z, roll, yaw, pitch].
        x2_pose (list): Pose of target [x, y, z, roll, yaw, pitch].
    Returns:
        np.ndarray: 4x4 Transformation Matrix.
    """
    # Convert degrees to radians
    x1_pose = [x1_pose[0], x1_pose[1], x1_pose[2], 
               math.radians(x1_pose[3]), math.radians(x1_pose[4]), math.radians(x1_pose[5])]
    x2_pose = [x2_pose[0], x2_pose[1], x2_pose[2], 
               math.radians(x2_pose[3]), math.radians(x2_pose[4]), math.radians(x2_pose[5])]

    # Rotation Matrix for x1 (Source)
    # Roll
    R_x1_r = np.array([[1, 0, 0], 
                       [0, math.cos(x1_pose[3]), -math.sin(x1_pose[3])], 
                       [0, math.sin(x1_pose[3]), math.cos(x1_pose[3])]])
    # Pitch
    R_x1_p = np.array([[math.cos(x1_pose[5]), 0, math.sin(x1_pose[5])], 
                       [0, 1, 0], 
                       [-math.sin(x1_pose[5]), 0, math.cos(x1_pose[5])]])
    # Yaw
    R_x1_y = np.array([[math.cos(x1_pose[4]), -math.sin(x1_pose[4]), 0], 
                       [math.sin(x1_pose[4]), math.cos(x1_pose[4]), 0], 
                       [0, 0, 1]])
    
    R_x1 = R_x1_y @ R_x1_p @ R_x1_r

    # Transformation Matrix x1 -> World
    T_x1_to_world = np.identity(4)
    T_x1_to_world[0:3, 0:3] = R_x1
    T_x1_to_world[0, 3] = x1_pose[0]
    T_x1_to_world[1, 3] = x1_pose[1]
    T_x1_to_world[2, 3] = x1_pose[2]

    # Rotation Matrix for x2 (Target)
    R_x2_r = np.array([[1, 0, 0], 
                       [0, math.cos(x2_pose[3]), -math.sin(x2_pose[3])], 
                       [0, math.sin(x2_pose[3]), math.cos(x2_pose[3])]])
    R_x2_p = np.array([[math.cos(x2_pose[5]), 0, math.sin(x2_pose[5])], 
                       [0, 1, 0], 
                       [-math.sin(x2_pose[5]), 0, math.cos(x2_pose[5])]])
    R_x2_y = np.array([[math.cos(x2_pose[4]), -math.sin(x2_pose[4]), 0], 
                       [math.sin(x2_pose[4]), math.cos(x2_pose[4]), 0], 
                       [0, 0, 1]])
    
    R_x2 = R_x2_y @ R_x2_p @ R_x2_r

    # Transformation Matrix x2 -> World
    T_x2_to_world = np.identity(4)
    T_x2_to_world[0:3, 0:3] = R_x2
    T_x2_to_world[0, 3] = x2_pose[0]
    T_x2_to_world[1, 3] = x2_pose[1]
    T_x2_to_world[2, 3] = x2_pose[2]

    # Transformation x1 -> x2 = (x2 -> World)^-1 @ (x1 -> World)
    T_x1_to_x2 = np.linalg.inv(T_x2_to_world) @ T_x1_to_world
    
    return T_x1_to_x2

def augment_lidar_and_targets(ego_lidar, rsu_lidar, target_point, waypoints, control):
    """
    Apply random rotation and scaling augmentation to LiDAR points and 
    corresponding trajectory targets during training.
    """
    # 1. Random Rotation
    if np.random.rand() < 0.5:
        # Rotation between -45 and 45 degrees
        angle = np.random.uniform(-np.pi / 4, np.pi / 4)
        c, s = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        
        # Apply to Ego LiDAR
        ego_lidar[:, :3] = np.dot(ego_lidar[:, :3], rotation_matrix.T)
        
        # Apply to RSU LiDAR
        rsu_lidar[:, :3] = np.dot(rsu_lidar[:, :3], rotation_matrix.T)
        
        # Apply to Target Point (2D)
        target_point = torch.tensor([
            target_point[0] * c - target_point[1] * s,
            target_point[0] * s + target_point[1] * c
        ])
        
        # Apply to Waypoints
        waypoints[:, 0] = waypoints[:, 0] * c - waypoints[:, 1] * s
        waypoints[:, 1] = waypoints[:, 0] * s + waypoints[:, 1] * c

    # 2. Random Scaling
    if np.random.rand() < 0.5:
        scale = np.random.uniform(0.95, 1.05)
        ego_lidar[:, :3] *= scale
        rsu_lidar[:, :3] *= scale
        target_point *= scale
        waypoints *= scale

    return ego_lidar, rsu_lidar, target_point, waypoints, control

# ==================================================================
# Dataset Class
# ==================================================================

class V2XVerse_TCP_Dataset(Dataset):
    """
    Dataset loader for CORTEX.
    Loads Ego-LiDAR, RSU-LiDAR, and navigation targets.
    Supports on-the-fly voxelization for PointPillars.
    """
    def __init__(self, raw_data_root, config, split='train', town_filter=None):
        self.root = Path(raw_data_root)
        self.config = config
        self.is_train = split == 'train'
        self.split = split
        
        # OpenCOOD configuration parameters
        self.lidar_range = config.opencood_params['lidar_range']
        self.voxel_size = config.opencood_params['voxel_size']
        self.max_points_per_voxel = config.opencood_params['max_points_per_voxel']
        self.max_voxel_num = config.opencood_params['max_voxel_train'] if self.is_train else config.opencood_params['max_voxel_test']

        # Load Dataset Index
        index_file = self.root / 'dataset_index.txt'
        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found at {index_file}. Run gen_index.py first.")
        
        with open(index_file, 'r') as f:
            all_lines = f.readlines()

        # Filter by towns
        if town_filter:
            self.sample_list = [line.strip() for line in all_lines if any(t in line for t in town_filter)]
        else:
            self.sample_list = [line.strip() for line in all_lines]

        # Flatten frames for direct indexing
        self.indexed_samples = []
        for line in self.sample_list:
            parts = line.split()
            if len(parts) < 2: continue
            
            # parts[0]: relative path to sequence, parts[1]: sequence length
            seq_path = parts[0]
            seq_len = int(parts[1])
            
            # Use frames with sufficient history/future
            # Skipping first few frames to ensure history exists if needed
            # and ensuring future frames exist for waypoint prediction (pred_len=4)
            start_frame = 5 
            end_frame = seq_len - 10 
            
            for i in range(start_frame, end_frame):
                self.indexed_samples.append({
                    'path': seq_path,
                    'frame_id': i
                })

        print(f"[{split.upper()}] Loaded {len(self.indexed_samples)} frames from {len(self.sample_list)} sequences.")

    def __len__(self):
        return len(self.indexed_samples)

    def _voxelize_lidar(self, points):
        """
        Convert point cloud to voxel features (compatible with OpenPCDet/PointPillars).
        """
        # 1. Filter out-of-range points
        points = mask_points_by_range(points, self.lidar_range)
        
        # 2. Voxelization logic (simplified for readability)
        # Calculate grid size
        grid_size = (np.array(self.lidar_range[3:6]) - np.array(self.lidar_range[0:3])) / np.array(self.voxel_size)
        grid_size = np.round(grid_size).astype(np.int64)
        
        # Shuffle points
        np.random.shuffle(points)
        
        # Quantize coords
        shifted_coord = points[:, :3] - np.array(self.lidar_range[:3])
        voxel_coords = np.floor(shifted_coord / np.array(self.voxel_size)).astype(np.int32)
        
        # Helper to create voxels (Naive implementation, optimized versions exist in spconv)
        # Using a dictionary for sparse storage
        voxel_dict = {}
        
        for i, coord in enumerate(voxel_coords):
            # Check bounds
            if np.any(coord < 0) or np.any(coord >= grid_size):
                continue
            
            k = tuple(coord)
            if k not in voxel_dict:
                if len(voxel_dict) >= self.max_voxel_num:
                    continue
                voxel_dict[k] = []
            
            if len(voxel_dict[k]) < self.max_points_per_voxel:
                voxel_dict[k].append(points[i])

        # Prepare tensors
        voxels = []
        coordinates = []
        num_points = []
        
        for k, v in voxel_dict.items():
            # Pad points to max_points_per_voxel
            pts = np.array(v)
            num = len(pts)
            padded_pts = np.zeros((self.max_points_per_voxel, 4), dtype=np.float32)
            padded_pts[:num, :] = pts
            
            voxels.append(padded_pts)
            # Coordinates format: (z, y, x) -> standard for PointPillars scatter
            coordinates.append([k[2], k[1], k[0]]) 
            num_points.append(num)
            
        if len(voxels) == 0:
            # Handle empty case
            return {
                'voxel_features': np.zeros((1, self.max_points_per_voxel, 4), dtype=np.float32),
                'voxel_coords': np.zeros((1, 3), dtype=np.int32),
                'voxel_num_points': np.zeros((1,), dtype=np.int32)
            }

        return {
            'voxel_features': np.array(voxels, dtype=np.float32),
            'voxel_coords': np.array(coordinates, dtype=np.int32),
            'voxel_num_points': np.array(num_points, dtype=np.int32)
        }

    def __getitem__(self, idx):
        try:
            sample_info = self.indexed_samples[idx]
            seq_path = self.root / sample_info['path']
            frame_id = sample_info['frame_id']

            # -------------------------------------
            # 1. Load Ego Vehicle Data
            # -------------------------------------
            # Load Measurements
            with open(seq_path / 'ego_vehicle_0/measurements' / f'{frame_id:04d}.json') as f:
                meas_data = json.load(f)
            
            # Load Ego Pose
            ego_pose = np.array([meas_data['x'], meas_data['y'], meas_data['z'], 
                                 meas_data['roll'], meas_data['yaw'], meas_data['pitch']])
            
            # Load Ego LiDAR
            lidar_file = seq_path / 'ego_vehicle_0/lidar' / f'{frame_id:04d}.npy'
            ego_lidar = np.load(lidar_file)
            ego_lidar = mask_ego_points(ego_lidar)

            # -------------------------------------
            # 2. Load RSU Data (Collaborative)
            # -------------------------------------
            # Assuming RSU_0 for simplicity (extendable to multi-RSU)
            rsu_path = seq_path / 'rsu_0'
            
            # Load RSU Pose (Static, but loaded for generality)
            with open(rsu_path / 'measurements' / f'{frame_id:04d}.json') as f:
                rsu_meas = json.load(f)
            rsu_pose = np.array([rsu_meas['x'], rsu_meas['y'], rsu_meas['z'], 
                                 rsu_meas['roll'], rsu_meas['yaw'], rsu_meas['pitch']])
            
            # Calculate Transformation Matrix: RSU -> Ego
            transformation_matrix = x1_to_x2(rsu_pose, ego_pose)
            
            # Load RSU LiDAR
            rsu_lidar_file = rsu_path / 'lidar' / f'{frame_id:04d}.npy'
            rsu_lidar = np.load(rsu_lidar_file)
            
            # Transform RSU points to Ego coordinate system
            # Homogeneous coordinates
            rsu_xyz = rsu_lidar[:, :3]
            ones = np.ones((rsu_xyz.shape[0], 1))
            rsu_xyz_h = np.hstack((rsu_xyz, ones))
            rsu_xyz_ego = (transformation_matrix @ rsu_xyz_h.T).T
            
            # Reconstruct RSU LiDAR with intensity
            rsu_lidar_transformed = np.zeros_like(rsu_lidar)
            rsu_lidar_transformed[:, :3] = rsu_xyz_ego[:, :3]
            rsu_lidar_transformed[:, 3] = rsu_lidar[:, 3]

            # -------------------------------------
            # 3. Targets & Measurements
            # -------------------------------------
            # Target Point (Goal) in Ego frame
            # (Simplified: assumes target is provided or calculated relative to ego)
            # In a real pipeline, this comes from the global planner.
            # Here we simulate a target point 20m ahead based on command/waypoints.
            
            # Future Waypoints (Ground Truth for TCP)
            waypoints_gt = []
            yaw_rad = math.radians(ego_pose[4])
            
            # Load future measurements for ground truth
            for i in range(1, self.config.tcp_params['pred_len'] + 1):
                future_file = seq_path / 'ego_vehicle_0/measurements' / f'{frame_id + i:04d}.json'
                if os.path.exists(future_file):
                    with open(future_file) as f:
                        future_meas = json.load(f)
                    future_pose = np.array([future_meas['x'], future_meas['y']])
                else:
                    # Fallback if end of sequence
                    future_pose = ego_pose[:2]
                
                # Global to Local transformation for waypoints
                diff = future_pose - ego_pose[:2]
                wx = diff[0] * math.cos(-yaw_rad) - diff[1] * math.sin(-yaw_rad)
                wy = diff[0] * math.sin(-yaw_rad) + diff[1] * math.cos(-yaw_rad)
                waypoints_gt.append([wx, wy])

            # Prepare Tensors
            target_point = np.array(waypoints_gt[-1]) # Use last waypoint as target approximation
            target_point = torch.tensor(target_point, dtype=torch.float32)
            waypoints_gt = torch.tensor(waypoints_gt, dtype=torch.float32)
            
            control_gt = torch.tensor([
                meas_data['throttle'], 
                meas_data['steer'], 
                meas_data['brake']
            ], dtype=torch.float32)

            measurements = torch.tensor([
                meas_data['speed'],
                target_point[0], target_point[1], # Target x, y
                meas_data.get('command', 2.0),    # Command (one-hot or int)
            ], dtype=torch.float32)
            
            # Augmentation (Training Only)
            if self.is_train:
                ego_lidar, rsu_lidar_transformed, target_point, waypoints_gt, control_gt = \
                    augment_lidar_and_targets(ego_lidar, rsu_lidar_transformed, target_point, waypoints_gt, control_gt)

            # Voxelization
            ego_lidar_dict = self._voxelize_lidar(ego_lidar)
            rsu_lidar_dict = self._voxelize_lidar(rsu_lidar_transformed)

            return {
                'ego_lidar_dict': ego_lidar_dict,
                'rsu_lidar_dict': rsu_lidar_dict,
                'transformation_matrix': torch.tensor(transformation_matrix, dtype=torch.float32),
                'measurements': measurements,
                'target_point': target_point,
                'waypoints': waypoints_gt,
                'control_gt': control_gt
            }

        except Exception as e:
            # print(f"Error loading frame {idx}: {e}")
            # traceback.print_exc()
            return None

# ==================================================================
# Collate Function
# ==================================================================

def tcp_collate_fn(batch):
    """
    Custom collate function to handle dictionary batching and filtering None samples.
    """
    # Filter failed samples
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None

    # Helper to collate voxel dicts (stacking features)
    def collate_voxel_dict(voxel_list):
        voxel_features = []
        voxel_coords = []
        voxel_num_points = []
        
        for i, v_dict in enumerate(voxel_list):
            voxel_features.append(torch.from_numpy(v_dict['voxel_features']))
            voxel_num_points.append(torch.from_numpy(v_dict['voxel_num_points']))
            
            # Add batch index to coordinates (required for PointPillars)
            # Original coords: (z, y, x) -> New coords: (batch_idx, z, y, x)
            coords = torch.from_numpy(v_dict['voxel_coords'])
            batch_idx = torch.full((coords.shape[0], 1), i, dtype=torch.int32)
            voxel_coords.append(torch.cat([batch_idx, coords], dim=1))
            
        return {
            'voxel_features': torch.cat(voxel_features, dim=0),
            'voxel_coords': torch.cat(voxel_coords, dim=0),
            'voxel_num_points': torch.cat(voxel_num_points, dim=0),
            'batch_size': len(batch)
        }

    ego_voxel_list = [b['ego_lidar_dict'] for b in batch]
    rsu_voxel_list = [b['rsu_lidar_dict'] for b in batch]

    return {
        'ego_lidar_dict': collate_voxel_dict(ego_voxel_list),
        'rsu_lidar_dict': collate_voxel_dict(rsu_voxel_list),
        'transformation_matrix': torch.stack([b['transformation_matrix'] for b in batch]),
        'measurements': torch.stack([b['measurements'] for b in batch]),
        'target_point': torch.stack([b['target_point'] for b in batch]),
        'waypoints': torch.stack([b['waypoints'] for b in batch]),
        'control_gt': torch.stack([b['control_gt'] for b in batch])
    }