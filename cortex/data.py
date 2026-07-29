"""
data.py - High-Performance Cross-Platform Data Pipeline for CORTEX
===================================================================
Official Dataset implementation for V2XVerse benchmark supporting
point-cloud voxelization, spatial coordinate alignment, lazy RAM caching,
and cross-platform execution (Windows/Linux).
"""

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

# Global process-level lazy RAM caches to reduce disk I/O across training epochs
_RSU_GEOMETRY_LAZY_CACHE: Dict[str, Any] = {}
_JSON_TRACKING_LAZY_CACHE: Dict[str, Any] = {}


def mask_points_by_range(points: np.ndarray, limit_range: List[float]) -> np.ndarray:
    """Filters 3D point cloud coordinates within a defined spatial bounding box.

    Args:
        points (np.ndarray): Input 3D point cloud array of shape (N, 3+C).
        limit_range (List[float]): Bounding box limits [x_min, y_min, z_min,
          x_max, y_max, z_max].

    Returns:
        np.ndarray: Filtered point cloud subset.
    """
    mask = (
        (points[:, 0] >= limit_range[0])
        & (points[:, 0] <= limit_range[3])
        & (points[:, 1] >= limit_range[1])
        & (points[:, 1] <= limit_range[4])
        & (points[:, 2] >= limit_range[2])
        & (points[:, 2] <= limit_range[5])
    )
    return points[mask]


def mask_ego_points(points: np.ndarray) -> np.ndarray:
    """Removes point cloud reflections originating from the ego-vehicle's own body.

    Args:
        points (np.ndarray): Input point cloud array (N, 3+C).

    Returns:
        np.ndarray: Point cloud with ego-vehicle footprint masked out.
    """
    mask = (
        (points[:, 0] < -1.5)
        | (points[:, 0] > 1.5)
        | (points[:, 1] < -1.0)
        | (points[:, 1] > 1.0)
    )
    return points[mask]


def apply_point_dropout(
    points: np.ndarray, dropout_rate: float = 0.2
) -> np.ndarray:
    """Applies random stochastic point dropout for data augmentation during training.

    Args:
        points (np.ndarray): Input point cloud array.
        dropout_rate (float): Probability threshold for point removal.

    Returns:
        np.ndarray: Augmented point cloud array.
    """
    if dropout_rate <= 0:
        return points
    keep_mask = np.random.uniform(0, 1, size=points.shape[0]) > dropout_rate
    return points[keep_mask]


def x1_to_x2(x1_pose: np.ndarray, x2_pose: np.ndarray) -> np.ndarray:
    """Computes the 4x4 rigid SE(3) transformation matrix mapping coordinates from

    Frame 1 to Frame 2 coordinate systems.

    Args:
        x1_pose (np.ndarray): Source pose vector [x, y, z, roll, pitch/yaw,
          yaw].
        x2_pose (np.ndarray): Target pose vector [x, y, z, roll, pitch/yaw,
          yaw].

    Returns:
        np.ndarray: 4x4 homogenous transformation matrix.
    """
    x1_world = np.eye(4)
    yaw1_rad = np.deg2rad(x1_pose[4])
    rot1 = np.array(
        [
            [np.cos(yaw1_rad), -np.sin(yaw1_rad), 0],
            [np.sin(yaw1_rad), np.cos(yaw1_rad), 0],
            [0, 0, 1],
        ]
    )
    x1_world[:3, :3] = rot1
    x1_world[:3, 3] = x1_pose[:3]

    x2_world = np.eye(4)
    yaw2_rad = np.deg2rad(x2_pose[4])
    rot2 = np.array(
        [
            [np.cos(yaw2_rad), -np.sin(yaw2_rad), 0],
            [np.sin(yaw2_rad), np.cos(yaw2_rad), 0],
            [0, 0, 1],
        ]
    )
    x2_world[:3, :3] = rot2
    x2_world[:3, 3] = x2_pose[:3]

    return np.linalg.inv(x2_world) @ x1_world


class V2XVerse_TCP_Dataset(Dataset):
    """V2XVerse Dataset loader tailored for the CORTEX cooperative driving architecture.

    Handles ego LiDAR, infrastructure (RSU) LiDAR alignment, kinematic measurements,
    and ground-truth future waypoint trajectory generation.
    """

    def __init__(
        self,
        raw_data_root: Union[str, Path],
        config: Any,
        split: str = 'train',
        town_filter: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.raw_data_root = Path(raw_data_root)
        self.config = config
        self.split = split
        self.is_train = split == 'train'
        self.dropout_rate = config.train_params.get('point_dropout_rate', 0.2)
        self.town_filter = town_filter

        index_file_path = self.raw_data_root / 'dataset_index.txt'
        if not index_file_path.exists():
            raise FileNotFoundError(
                f'Dataset index file not found at: {index_file_path}'
            )

        with open(index_file_path, 'r', encoding='utf-8') as f:
            all_samples_info = f.readlines()

        self.routes = all_samples_info
        self.flat_index = self._build_flat_index()
        print(
            f"🚀 [CORTEX Pipeline] Dataset '{split}' initialized with"
            f' {len(self.flat_index)} samples.'
        )

    def _create_pose_from_json(
        self, data: Optional[Dict[str, Any]]
    ) -> Optional[np.ndarray]:
        """Extracts spatial 6-DoF pose vector from measurement JSON metadata."""
        if data is None or 'lidar_pose_x' not in data:
            return None
        return np.array([
            data['lidar_pose_x'],
            data['lidar_pose_y'],
            data['lidar_pose_z'],
            0.0,
            math.degrees(data['theta']),
            0.0,
        ])

    def _voxelize_lidar(self, lidar_points: np.ndarray) -> Dict[str, np.ndarray]:
        """Fast 1D linear mapping voxelization pipeline for PointPillar backbone ingestion.

        Args:
            lidar_points (np.ndarray): Array of 3D point coordinates (N, 4).

        Returns:
            Dict[str, np.ndarray]: Dictionary containing 'voxel_features',
            'voxel_num_points', and 'voxel_coords'.
        """
        if lidar_points.shape[0] == 0:
            return {
                'voxel_features': np.zeros((0, 32, 4), dtype=np.float32),
                'voxel_num_points': np.zeros(0, dtype=np.int32),
                'voxel_coords': np.zeros((0, 3), dtype=np.int32),
            }

        if self.is_train:
            lidar_points = apply_point_dropout(
                lidar_points, self.dropout_rate
            )

        voxel_size = np.array(self.config.opencood_params['voxel_size'])
        pc_range = np.array(self.config.opencood_params['point_cloud_range'])
        grid_size_np = np.array(
            self.config.opencood_params['point_pillar_scatter']['grid_size']
        )

        voxel_coords = np.floor(
            (lidar_points[:, :3] - pc_range[:3]) / voxel_size
        ).astype(np.int32)
        mask = np.all(
            (voxel_coords >= 0) & (voxel_coords < grid_size_np[:3]), axis=1
        )
        lidar_points = lidar_points[mask]
        voxel_coords = voxel_coords[mask]

        if voxel_coords.shape[0] == 0:
            return {
                'voxel_features': np.zeros((0, 32, 4), dtype=np.float32),
                'voxel_num_points': np.zeros(0, dtype=np.int32),
                'voxel_coords': np.zeros((0, 3), dtype=np.int32),
            }

        # Vectorized 1D linear spatial index computation for fast pillar gathering
        nx, ny = int(grid_size_np[0]), int(grid_size_np[1])
        flat_idx = voxel_coords[:, 0] * ny + voxel_coords[:, 1]

        unique_flat_idx, inverse_indices = np.unique(
            flat_idx, return_inverse=True
        )
        max_points = 32

        idx_sort = np.argsort(inverse_indices)
        lidar_sorted = lidar_points[idx_sort]
        inverse_sorted = inverse_indices[idx_sort]

        diff_m = np.where(inverse_sorted[:-1] != inverse_sorted[1:])[0] + 1
        splits = np.r_[0, diff_m, len(inverse_sorted)]

        group_starts = np.repeat(splits[:-1], np.diff(splits))
        rank = np.arange(len(inverse_sorted)) - group_starts

        keep_mask = rank < max_points
        filtered_lidar = lidar_sorted[keep_mask]
        filtered_inverse = inverse_sorted[keep_mask]
        filtered_rank = rank[keep_mask]

        voxel_features = np.zeros(
            (len(unique_flat_idx), max_points, 4), dtype=np.float32
        )
        voxel_features[filtered_inverse, filtered_rank, :] = filtered_lidar[
            :, :4
        ]
        voxel_num_points = np.minimum(np.diff(splits), max_points).astype(
            np.int32
        )

        # Reconstruct 3D grid coordinates [z, y, x] for PointPillar Scatter module
        unique_x = unique_flat_idx // ny
        unique_y = unique_flat_idx % ny
        unique_z = np.zeros_like(unique_x)
        unique_coords = np.stack([unique_z, unique_y, unique_x], axis=1)

        return {
            'voxel_features': voxel_features,
            'voxel_num_points': voxel_num_points,
            'voxel_coords': unique_coords,
        }

    def _build_flat_index(self) -> List[Dict[str, Any]]:
        """Parses dataset index text file and constructs sample lookup entries."""
        flat_index = []
        pred_len = self.config.tcp_params['pred_len']
        for line in self.routes:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            route_path_str, seq_len = parts[0], int(parts[1])

            if self.town_filter is not None:
                if not any(
                    town.lower() in route_path_str.lower()
                    for town in self.town_filter
                ):
                    continue

            for frame_id in range(1, seq_len - (pred_len + 5)):
                flat_index.append(
                    {'route_path_str': route_path_str, 'frame_id': frame_id}
                )
        return flat_index

    def __len__(self) -> int:
        return len(self.flat_index)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        try:
            sample = self.flat_index[idx]
            route_path_str = sample['route_path_str']

            # Cross-platform Path normalization (Windows & Linux compatible)
            if 'weather-' in route_path_str:
                idx_w = route_path_str.find('weather-')
                clean_rel_path = route_path_str[idx_w:].replace('\\', '/')
                route_path = self.raw_data_root / Path(clean_rel_path)
            else:
                route_path = self.raw_data_root / Path(
                    route_path_str.replace('\\', '/')
                )

            frame_id = sample['frame_id']
            ego_path = route_path / 'ego_vehicle_0'

            # Lazy RAM cache wrapper for metadata JSON parsing
            if route_path_str not in _JSON_TRACKING_LAZY_CACHE:
                _JSON_TRACKING_LAZY_CACHE[route_path_str] = {}

            def load_json_cached(file_path: Path) -> Optional[Dict[str, Any]]:
                f_key = str(file_path)
                if f_key not in _JSON_TRACKING_LAZY_CACHE[route_path_str]:
                    if file_path.exists():
                        with open(file_path, 'r', encoding='utf-8') as json_f:
                            _JSON_TRACKING_LAZY_CACHE[route_path_str][f_key] = (
                                json.load(json_f)
                            )
                    else:
                        _JSON_TRACKING_LAZY_CACHE[route_path_str][f_key] = None
                return _JSON_TRACKING_LAZY_CACHE[route_path_str][f_key]

            ego_meas_file = ego_path / 'measurements' / f'{frame_id:04d}.json'
            meas_data = load_json_cached(ego_meas_file)
            if meas_data is None:
                return None

            ego_pose = self._create_pose_from_json(meas_data)
            yaw_rad = meas_data['theta']

            ego_lidar_file = ego_path / 'lidar' / f'{frame_id:04d}.npy'
            if not ego_lidar_file.exists():
                return None

            ego_lidar = np.load(ego_lidar_file)
            ego_lidar[:, 1] *= -1.0  # Coordinate axis orientation flip
            ego_lidar = mask_points_by_range(
                mask_ego_points(ego_lidar),
                self.config.opencood_params['point_cloud_range'],
            )

            # Lazy RAM cache for static Roadside Unit (RSU) spatial topology
            if route_path_str not in _RSU_GEOMETRY_LAZY_CACHE:
                _RSU_GEOMETRY_LAZY_CACHE[route_path_str] = []
                rsu_dirs = [
                    d
                    for d in route_path.iterdir()
                    if d.is_dir() and 'rsu' in d.name.lower()
                ]
                for rf in rsu_dirs:
                    for first_json in (rf / 'measurements').glob('*.json'):
                        try:
                            with open(first_json, 'r', encoding='utf-8') as r_f:
                                r_meas = json.load(r_f)
                            r_pose = self._create_pose_from_json(r_meas)
                            if r_pose is not None:
                                _RSU_GEOMETRY_LAZY_CACHE[
                                    route_path_str
                                ].append({'path': rf, 'pose': r_pose})
                                break
                        except Exception:
                            continue

            rsu_info_list = _RSU_GEOMETRY_LAZY_CACHE[route_path_str]
            best_rsu, min_d, best_rp = None, float('inf'), None

            if rsu_info_list:
                ego_xy = ego_pose[:2]
                for rsu_info in rsu_info_list:
                    dist = np.linalg.norm(ego_xy - rsu_info['pose'][:2])
                    if dist < min_d:
                        min_d = dist
                        best_rsu = rsu_info['path']
                        best_rp = rsu_info['pose']

            if best_rsu is not None:
                rsu_lidar_file = best_rsu / 'lidar' / f'{frame_id:04d}.npy'
                if rsu_lidar_file.exists():
                    rsu_lidar_raw = np.load(rsu_lidar_file)
                    tm_4x4 = x1_to_x2(best_rp, ego_pose)
                    rsu_homo = np.pad(
                        rsu_lidar_raw[:, :3],
                        ((0, 0), (0, 1)),
                        constant_values=1,
                    )
                    rsu_ego_frame = (tm_4x4 @ rsu_homo.T).T[:, :3]

                    rsu_lidar_final = np.hstack(
                        (rsu_ego_frame, rsu_lidar_raw[:, 3:])
                    )
                    rsu_lidar_final[:, 1] *= -1.0
                    transformation_matrix = tm_4x4[:2, [0, 1, 3]].astype(
                        np.float32
                    )
                else:
                    rsu_lidar_final = np.zeros((0, 4), dtype=np.float32)
                    transformation_matrix = np.eye(4)[:2, [0, 1, 3]].astype(
                        np.float32
                    )
            else:
                rsu_lidar_final = np.zeros((0, 4), dtype=np.float32)
                transformation_matrix = np.eye(4)[:2, [0, 1, 3]].astype(
                    np.float32
                )

            speed = meas_data['speed']
            vx_local = speed
            vy_local = 0.0
            omega = 0.0
            should_brake = float(meas_data.get('should_brake', 0))

            # Finite derivative approximation for angular yaw rate (omega) and lateral velocity (vy)
            try:
                prev_ego_meas_file = (
                    ego_path / 'measurements' / f'{(frame_id - 1):04d}.json'
                )
                prev_meas_data = load_json_cached(prev_ego_meas_file)
                if prev_meas_data is not None:
                    dt = 0.1
                    dx_global = meas_data['x'] - prev_meas_data['x']
                    dy_global = meas_data['y'] - prev_meas_data['y']
                    vy_local = (
                        -dx_global * np.sin(yaw_rad)
                        + dy_global * np.cos(yaw_rad)
                    ) / dt
                    delta_theta = math.atan2(
                        math.sin(yaw_rad - prev_meas_data['theta']),
                        math.cos(yaw_rad - prev_meas_data['theta']),
                    )
                    omega = delta_theta / dt
            except Exception:
                pass

            target_world = np.array(
                [meas_data['near_node_x'], meas_data['near_node_y']]
            )
            delta = target_world - ego_pose[:2]
            target_local = np.array([
                delta[0] * np.cos(yaw_rad) + delta[1] * np.sin(yaw_rad),
                -delta[0] * np.sin(yaw_rad) + delta[1] * np.cos(yaw_rad),
            ])

            cmd_vec = np.zeros(6)
            cmd_vec[meas_data['command'] - 1] = 1.0

            # 11-channel heterogeneous measurement vector construction
            measurements = torch.tensor(
                np.concatenate([
                    target_local,
                    [speed / 12.0],
                    [vx_local, vy_local],
                    cmd_vec,
                ]),
                dtype=torch.float32,
            )

            waypoints_gt = []
            for i in range(1, self.config.tcp_params['pred_len'] + 1):
                f_file = ego_path / 'measurements' / f'{frame_id + i:04d}.json'
                f_meas = load_json_cached(f_file)
                if f_meas is None:
                    return None
                f_pose = self._create_pose_from_json(f_meas)
                d_f = f_pose[:2] - ego_pose[:2]
                waypoints_gt.append([
                    d_f[0] * np.cos(yaw_rad) + d_f[1] * np.sin(yaw_rad),
                    -d_f[0] * np.sin(yaw_rad) + d_f[1] * np.cos(yaw_rad),
                ])

            if rsu_lidar_final.shape[0] > 0:
                rsu_lidar_final = mask_points_by_range(
                    rsu_lidar_final,
                    self.config.opencood_params['point_cloud_range'],
                )

            waypoints_gt_np = np.array(waypoints_gt, dtype=np.float32)

            return {
                'ego_lidar_dict': self._voxelize_lidar(ego_lidar),
                'rsu_lidar_dict': self._voxelize_lidar(rsu_lidar_final),
                'transformation_matrix': transformation_matrix,
                'measurements': measurements,
                'target_point': torch.tensor(
                    target_local, dtype=torch.float32
                ),
                'waypoints_gt': torch.tensor(
                    waypoints_gt_np, dtype=torch.float32
                ),
                'control_gt': torch.tensor(
                    [
                        meas_data['throttle'],
                        meas_data['steer'],
                        meas_data['brake'],
                    ],
                    dtype=torch.float32,
                ),
                'communication_delay': torch.tensor(
                    random.uniform(0.05, 0.30)
                )
                if self.is_train
                else torch.tensor(0.1),
                'physical_omega': torch.tensor(omega, dtype=torch.float32),
                'physical_should_brake': torch.tensor(
                    should_brake, dtype=torch.float32
                ),
            }
        except Exception:
            return None


def tcp_collate_fn(batch: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """Custom collation function to assemble batched sparse voxel dictionaries

    and multi-modal tensor measurements for DataLoader workers.
    """
    batch_filtered = [d for d in batch if d is not None]
    if not batch_filtered:
        return None

    res: Dict[str, List[Any]] = {
        k: []
        for k in [
            'measurements',
            'target_point',
            'waypoints_gt',
            'control_gt',
            'transformation_matrix',
            'communication_delay',
            'physical_omega',
            'physical_should_brake',
        ]
    }
    ego_vox, ego_num, ego_coords = [], [], []
    rsu_vox, rsu_num, rsu_coords = [], [], []

    for i, data in enumerate(batch_filtered):
        for k in res:
            res[k].append(data[k])

        e_dict = data['ego_lidar_dict']
        ego_vox.append(e_dict['voxel_features'])
        ego_num.append(e_dict['voxel_num_points'])
        ego_coords.append(
            np.pad(
                e_dict['voxel_coords'], ((0, 0), (1, 0)), constant_values=i
            )
        )

        r_dict = data['rsu_lidar_dict']
        if r_dict['voxel_coords'].shape[0] > 0:
            rsu_vox.append(r_dict['voxel_features'])
            rsu_num.append(r_dict['voxel_num_points'])
            rsu_coords.append(
                np.pad(
                    r_dict['voxel_coords'], ((0, 0), (1, 0)), constant_values=i
                )
            )

    final_batch = {
        k: torch.stack(v)
        if k != 'transformation_matrix'
        else torch.from_numpy(np.array(v))
        for k, v in res.items()
    }

    final_batch['ego_lidar_dict'] = {
        'voxel_features': torch.from_numpy(np.concatenate(ego_vox)),
        'voxel_num_points': torch.from_numpy(np.concatenate(ego_num)),
        'voxel_coords': torch.from_numpy(np.concatenate(ego_coords)),
        'batch_size': len(batch_filtered),
    }

    if len(rsu_vox) > 0:
        final_batch['rsu_lidar_dict'] = {
            'voxel_features': torch.from_numpy(np.concatenate(rsu_vox)),
            'voxel_num_points': torch.from_numpy(np.concatenate(rsu_num)),
            'voxel_coords': torch.from_numpy(np.concatenate(rsu_coords)),
            'batch_size': len(batch_filtered),
        }
    else:
        final_batch['rsu_lidar_dict'] = None

    return final_batch