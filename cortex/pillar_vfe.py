"""
Pillar VFE (Voxel Feature Encoding) Module for CORTEX Framework.
Applies localized augmentation and Pillar Feature Network (PFN) transformations 
to convert raw 3D point cloud coordinate distributions into pillar embeddings.

Ref: CORTEX Paper - Section III-C, Equations (5) and (6)
Credits: Adapted from OpenPCDet with CORTEX multi-modal extensions.
"""

from typing import Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class PFNLayer(nn.Module):
    """
    Pillar Feature Network (PFN) Layer: Maps augmented point vectors into 
    dense embedding space via Linear -> Norm -> ReLU -> Max Pooling over intra-pillar points.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 use_norm: bool = True,
                 last_layer: bool = False) -> None:
        super().__init__()

        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Augmented pillar point tensor [M, Max_Points, C_in].
        Returns:
            x_max or x_concatenated: Permutation-invariant pillar embeddings [M, C_out].
        """
        if inputs.shape[0] > self.part:
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [
                self.linear(inputs[num_part * self.part:(num_part + 1) * self.part])
                for num_part in range(num_parts + 1)
            ]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)

        # Permute for 1D batch normalization along feature channels
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        x = F.relu(x)
        
        # Intra-pillar Max Pooling for point density permutation invariance (Eq. 6)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            return torch.cat([x, x_repeat], dim=2)


class PillarVFE(nn.Module):
    """
    Pillar Voxel Feature Encoder: Enriches raw 3D points within vertical columnar bins (pillars)
    with arithmetic cluster means, geometric cell offsets, and Euclidean range profiles.
    """
    def __init__(self, 
                 model_cfg: Dict[str, Any], 
                 num_point_features: int, 
                 voxel_size: List[float],
                 point_cloud_range: List[float]) -> None:
        super().__init__()
        self.model_cfg = model_cfg

        self.use_norm = self.model_cfg['use_norm']
        self.with_distance = self.model_cfg['with_distance']
        self.use_absolute_xyz = self.model_cfg['use_absolute_xyz']

        # Determine total augmented feature dimension (Eq. 5 in paper)
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters: List[int] = self.model_cfg['num_filters']
        assert len(self.num_filters) > 0, "num_filters configuration cannot be empty"
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm,
                         last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x, self.voxel_y, self.voxel_z = voxel_size[0], voxel_size[1], voxel_size[2]
        self.x_offset = self.voxel_x / 2.0 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2.0 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2.0 + point_cloud_range[2]

    def get_output_feature_dim(self) -> int:
        return self.num_filters[-1]

    @staticmethod
    def get_paddings_indicator(actual_num: torch.Tensor, max_num: int, axis: int = 0) -> torch.Tensor:
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num_tensor = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        return actual_num.int() > max_num_tensor

    def forward(self, batch_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            batch_dict: Dictionary containing:
                - 'voxel_features': Sparse point clouds [M, Max_Points, C]
                - 'voxel_num_points': Active point count per pillar [M]
                - 'voxel_coords': Voxel spatial coordinates [M, 4]
        Returns:
            batch_dict: Dictionary updated with 'pillar_features' [M, C_out]
        """
        voxel_features = batch_dict['voxel_features']
        voxel_num_points = batch_dict['voxel_num_points']
        coords = batch_dict['voxel_coords']

        # 1. Compute arithmetic mean coordinate across intra-pillar points (f_cluster)
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / \
                      voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean

        # 2. Compute coordinate offset from geometric midpoint of cell (f_center)
        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (
            coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset
        )
        f_center[:, :, 1] = voxel_features[:, :, 1] - (
            coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset
        )
        f_center[:, :, 2] = voxel_features[:, :, 2] - (
            coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset
        )

        # 3. Concatenate feature channels to build augmented point vector p_aug
        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
            
        features = torch.cat(features, dim=-1)

        # 4. Mask padded zero-tokens in non-full pillars
        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask

        # 5. Pass through PFN layers
        for pfn in self.pfn_layers:
            features = pfn(features)
            
        features = features.squeeze(1)
        batch_dict['pillar_features'] = features

        return batch_dict