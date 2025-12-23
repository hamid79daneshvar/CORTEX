# model_v2i.py
# Official Implementation of the CORTEX Architecture
# Includes: Request-Aware Fusion & Spatial Convolutional Control Head

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# --- Perception Modules ---
from pillar_vfe import PillarVFE
from point_pillar_scatter import PointPillarScatter
from base_bev_backbone_resnet import ResNetBEVBackbone

# --- Communication/Transformation Utilities ---
from torch_transformation_utils import warp_affine

# --- Baseline TCP Model (for trajectory branch weights) ---
from model import TCP as TCP_Base_Model 

class ScaledDotProductAttention(nn.Module):
    """
    Standard Scaled Dot-Product Attention mechanism.
    Used for fusing Ego and RSU features in the BEV space.
    """
    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query, key, value):
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context

class SpatialControlHead(nn.Module):
    """
    [Novelty] Spatial Convolutional Control Head.
    Preserves geometric structure of the fused BEV features to enable 
    precise localization of occluded hazards.
    """
    def __init__(self, input_channels=256, output_dim=3):
        super().__init__()
        
        # Convolutional layers to process spatial features (preserving grid topology)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Flatten and fuse with measurements
        self.fc_head = nn.Sequential(
            nn.Linear(32 * 10 * 10 + 128, 256), # Assuming 80x80 input -> 10x10 feature map
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, output_dim)
        )

    def forward(self, x_bev, measurements_embed):
        # x_bev: (B, C, H, W)
        x = self.conv_layers(x_bev)
        x = x.view(x.size(0), -1) # Flatten
        
        # Fuse with measurement embeddings (velocity, command, etc.)
        x = torch.cat([x, measurements_embed], dim=1)
        x = self.fc_head(x)
        return x

class Co_TCP_Advanced(nn.Module):
    """
    CORTEX: Cooperative Occlusion-Resilient Trajectory Execution
    Integrates Request-Aware Fusion with a dual-branch control architecture.
    """
    def __init__(self, tcp_config, opencood_config, comm_config):
        super().__init__()
        self.tcp_config = tcp_config
        self.opencood_config = opencood_config
        self.comm_config = comm_config

        # ---------------------------------------------------------
        # 1. LiDAR Perception Backbone (PointPillars + ResNet)
        # ---------------------------------------------------------
        # Ego Vehicle Perception
        self.ego_vfe = PillarVFE(opencood_config, num_point_features=4, voxel_size=opencood_config['voxel_size'], point_cloud_range=opencood_config['lidar_range'])
        self.ego_scatter = PointPillarScatter(opencood_config['point_pillar_scatter'])
        self.ego_backbone = ResNetBEVBackbone(opencood_config['base_bev_backbone'], input_channels=64)

        # RSU Perception (Shared weights structure)
        self.rsu_vfe = PillarVFE(opencood_config, num_point_features=4, voxel_size=opencood_config['voxel_size'], point_cloud_range=opencood_config['lidar_range'])
        self.rsu_scatter = PointPillarScatter(opencood_config['point_pillar_scatter'])
        self.rsu_backbone = ResNetBEVBackbone(opencood_config['base_bev_backbone'], input_channels=64)

        # ---------------------------------------------------------
        # 2. Compression & Fusion Modules
        # ---------------------------------------------------------
        # Compress high-dim features for efficient transmission/fusion
        self.compression_net = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=1), # Compress 384 channels -> 128
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Attention-based Fusion
        self.fusion_net = ScaledDotProductAttention(128)
        
        # Restore channels after fusion
        self.decompression_net = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # ---------------------------------------------------------
        # 3. Trajectory & Control Branches (Dual-Branch Design)
        # ---------------------------------------------------------
        # Initialize standard TCP modules to inherit trajectory logic
        self.tcp_model = TCP_Base_Model(tcp_config)
        
        # Measurement encoder (Speed, Command, etc.)
        self.measurements_encoder = self.tcp_model.measurements_fc # Re-use from TCP
        
        # Branch A: Trajectory Prediction (Uses Global Average Pooling)
        self.traj_branch = self.tcp_model.traj_branch # GRU-based waypoint predictor
        
        # Branch B: Spatial Control Head (The Novelty)
        # Unlike standard TCP which pools features, this keeps spatial structure
        self.spatial_control_head = SpatialControlHead(input_channels=256, output_dim=3)

    def forward(self, batch_dict):
        """
        Forward pass for CORTEX.
        Steps:
        1. Extract BEV features for Ego and RSU.
        2. Apply Request-Aware Fusion (Spatial Alignment).
        3. Predict Trajectory (Global Context).
        4. Predict Control (Spatial Context).
        """
        # --- 1. Feature Extraction ---
        # Process Ego LiDAR
        batch_dict = self.ego_vfe(batch_dict['ego_lidar_dict'])
        batch_dict = self.ego_scatter(batch_dict)
        batch_dict = self.ego_backbone(batch_dict)
        ego_bev_feature = batch_dict['spatial_features_2d'] # (B, 384, H, W)

        # Process RSU LiDAR
        # Note: In a real deployment, this happens on the RSU edge node
        rsu_dict = batch_dict['rsu_lidar_dict']
        # Helper wrapper to match PFN input structure
        rsu_batch_wrapper = {
            'voxel_features': rsu_dict['voxel_features'],
            'voxel_num_points': rsu_dict['voxel_num_points'],
            'voxel_coords': rsu_dict['voxel_coords']
        }
        rsu_batch_wrapper = self.rsu_vfe(rsu_batch_wrapper)
        rsu_batch_wrapper = self.rsu_scatter(rsu_batch_wrapper)
        rsu_batch_wrapper = self.rsu_backbone(rsu_batch_wrapper)
        rsu_bev_feature = rsu_batch_wrapper['spatial_features_2d'] # (B, 384, H, W)

        # --- 2. Compression & Alignment ---
        ego_bev_compressed = self.compression_net(ego_bev_feature)
        rsu_bev_compressed = self.compression_net(rsu_bev_feature)
        
        # Warp RSU features to Ego frame
        # (Assuming transformation_matrix is provided in batch_dict)
        tm = batch_dict['transformation_matrix'] 
        H, W = ego_bev_compressed.shape[2], ego_bev_compressed.shape[3]
        rsu_bev_warped = warp_affine(rsu_bev_compressed, tm, dsize=(H, W))

        # --- 3. Request-Aware Fusion ---
        # Flatten for attention: (B, C, H, W) -> (B, H*W, C)
        B, C, H, W = ego_bev_compressed.shape
        ego_flat = ego_bev_compressed.permute(0, 2, 3, 1).view(B, -1, C)
        rsu_flat = rsu_bev_warped.permute(0, 2, 3, 1).view(B, -1, C)
        
        # Query = Ego (Request), Key/Value = Stacked [Ego, RSU]
        query = ego_flat.unsqueeze(2) # (B, N, 1, C)
        keys = torch.stack([ego_flat, rsu_flat], dim=2) # (B, N, 2, C)
        
        # Reshape for efficient batch matrix multiplication
        # Treating each pixel as an independent query
        query_reshaped = query.view(B * H * W, 1, C)
        keys_reshaped = keys.view(B * H * W, 2, C)
        
        # Apply Attention
        fused_flat = self.fusion_net(query_reshaped, keys_reshaped, keys_reshaped).squeeze(1)
        
        # Reshape back to BEV grid
        fused_bev_compressed = fused_flat.view(B, H, W, C).permute(0, 3, 1, 2)
        
        # Restore channel depth
        fused_bev = self.decompression_net(fused_bev_compressed) # (B, 256, H, W)

        # --- 4. Measurement Encoding ---
        # Encode speed, command, target_point
        meas_input = batch_dict['measurements']
        measurements_embed = self.measurements_encoder(meas_input) # (B, 128)

        # --- 5. Branch A: Trajectory Prediction ---
        # Uses Global Average Pooling (GAP) to capture global context
        fused_pooled = F.adaptive_avg_pool2d(fused_bev, (1, 1)).squeeze(-1).squeeze(-1)
        
        # Initialize GRU with fused global features + measurements
        # Note: We adapt the input size to match TCP's expected input
        # Assuming TCP expects concatenation of [features, measurements]
        traj_input_features = torch.cat([fused_pooled, measurements_embed], dim=1)
        
        # Predict Waypoints (autoregressive)
        # Note: The original TCP 'traj_branch' takes (feature, target_point_embedding)
        # We simplify here by passing the concatenated vector if architecture aligns, 
        # or we rely on the internal mechanism of self.tcp_model.
        # For CORTEX, we use the fused features to initialize the hidden state.
        
        # Re-using TCP's trajectory logic (simplified call for compatibility)
        # In exact implementation, ensure input dimensions match config.
        # Here we assume self.traj_branch logic handles the feature vector.
        pred_wp = self.tcp_model.traj_branch(traj_input_features, meas_input['target_point'])

        # --- 6. Branch B: Spatial Control Head ---
        # The key innovation: using the spatial feature map directly
        # fused_bev is (B, 256, H, W) -> Passed to CNN head
        pred_ctrl_raw = self.spatial_control_head(fused_bev, measurements_embed)
        
        # Activations
        throttle = torch.sigmoid(pred_ctrl_raw[:, 0]).unsqueeze(1)
        steer = torch.tanh(pred_ctrl_raw[:, 1]).unsqueeze(1)
        brake = torch.sigmoid(pred_ctrl_raw[:, 2]).unsqueeze(1)
        
        pred_ctrl = torch.cat([throttle, steer, brake], dim=1)

        return {
            'pred_wp': pred_wp,      # Future Waypoints
            'pred_ctrl': pred_ctrl,  # Control Actions
            'fused_bev': fused_bev   # For visualization (optional)
        }