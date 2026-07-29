"""
PointPillar Scatter Module for CORTEX Architecture.
Converts 1D sparse pillar feature vectors into a structured 2D Bird's-Eye-View (BEV)
pseudo-image canvas tensor [B, C, H, W] via GPU-accelerated parallel index mapping.

Ref: CORTEX Paper - Section III-C (LiDAR-Based BEV Feature Extraction)
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn


class PointPillarScatter(nn.Module):
    """
    Scatters 1D pillar features back into Cartesian grid positions to generate a 
    structured 2D spatial pseudo-image tensor X_pseudo in R^{B x C x H x W}.
    """
    def __init__(self, model_cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features: int = self.model_cfg['num_features']
        self.nx, self.ny, self.nz = self.model_cfg['grid_size']
        assert self.nz == 1, f"PointPillar Scatter expects 2D BEV grid with nz=1, got nz={self.nz}"

    def forward(self, batch_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            batch_dict: Dictionary containing:
                - 'pillar_features': Tensor [M, C] extracted by PillarVFE.
                - 'voxel_coords': Tensor [M, 4] formatted as [batch_idx, z_idx, y_idx, x_idx].
                - 'batch_size': Optional int defining explicit batch size.
        Returns:
            batch_dict: Updated dictionary containing 'spatial_features' [B, C, H, W].
        """
        pillar_features = batch_dict['pillar_features']
        coords = batch_dict['voxel_coords']
        
        # Determine explicit batch size dynamically
        if 'batch_size' in batch_dict:
            batch_size = batch_dict['batch_size']
        else:
            batch_size = coords[:, 0].max().int().item() + 1 if coords.shape[0] > 0 else 0

        # Handle empty batch or missing point cloud edge case safely
        if coords.shape[0] == 0:
            batch_spatial_features = torch.zeros(
                batch_size, 
                self.num_bev_features * self.nz, 
                self.ny, 
                self.nx, 
                dtype=pillar_features.dtype, 
                device=pillar_features.device
            )
            batch_dict['spatial_features'] = batch_spatial_features
            return batch_dict

        # Allocate flat 1D memory canvas on GPU for parallel vector scatter
        flat_grid_size = self.nz * self.nx * self.ny
        spatial_feature = torch.zeros(
            batch_size * flat_grid_size,
            self.num_bev_features,
            dtype=pillar_features.dtype,
            device=pillar_features.device
        )
        
        # Compute global linear memory index: (batch_idx * flat_size) + (y_idx * nx) + x_idx
        linear_indices = (coords[:, 0].long() * flat_grid_size) + \
                         (coords[:, 2].long() * self.nx) + \
                         (coords[:, 3].long())
        
        # Parallel GPU scatter mapping without Python loops
        spatial_feature[linear_indices] = pillar_features
        
        # Reshape flat tensor into standard 2D BEV spatial format [B, C, H, W]
        spatial_feature = spatial_feature.view(
            batch_size, flat_grid_size, self.num_bev_features
        ).permute(0, 2, 1)
        
        batch_spatial_features = spatial_feature.view(
            batch_size, self.num_bev_features * self.nz, self.ny, self.nx
        )
        
        batch_dict['spatial_features'] = batch_spatial_features
        return batch_dict