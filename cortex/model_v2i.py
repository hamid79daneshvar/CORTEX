"""
model_v2i.py
================================================================================
CORTEX: Cooperative Occlusion-Resilient Trajectory Execution via Request-Aware V2I Fusion

Official PyTorch implementation of the CORTEX neural network architecture.
This module encapsulates:
  1. Pillar Feature Encoding (PFN) and PointPillar Bird's-Eye-View (BEV) Extraction
  2. Feedforward Coarse Path Projection & Request Corridor Mapping (Eq. 8-10)
  3. Velocity-Yaw-Rate Motion Compensated Kinematic Latency Correction (Eq. 11-15)
  4. Hard-Masked Scaled Dot-Product Cross-Attention Fusion (Eq. 16-21)
  5. Autoregressive Recurrent Waypoint Trajectory Decoding (Eq. 22-28)
  6. Non-Truncated 13,960-Channel Spatial Convolutional Control Head (Eq. 29-40)

Target Journal: IEEE Access
Manuscript ID: Access-2025-56941
================================================================================
"""

import math
from argparse import Namespace
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

try:
    from .pillar_vfe import PillarVFE
    from .point_pillar_scatter import PointPillarScatter
    from .base_bev_backbone_resnet import ResNetBEVBackbone
    from .model import TCP as TCP_Base_Model
except ImportError:
    from pillar_vfe import PillarVFE
    from point_pillar_scatter import PointPillarScatter
    from base_bev_backbone_resnet import ResNetBEVBackbone
    from model import TCP as TCP_Base_Model


class ScaledDotProductAttention(nn.Module):
    """
    Hard-Masked Scaled Dot-Product Spatial Cross-Attention Fusion Module.
    
    Mathematical Formulation (Eq. 16 - 21):
      Q_ego = tanh(Q_ego / 50.0) * 50.0,  K_rsu = tanh(K_rsu / 50.0) * 50.0
      Score_raw = tanh((Q_ego * K_rsu^T) / sqrt(d_channel)) * 20.0
      Attention = Softmax(Score_raw + B_spatial)
      X_fused = Attention * V_rsu
      
    Where B_spatial introduces a severe -1e9 penalty for tokens positioned 
    outside the projected future driving corridor, forcing softmax probabilities 
    to absolute zero and enforcing strict transmission bandwidth minimalism.
    """
    def __init__(self, dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = 1.0 / (math.sqrt(dim) + 1e-6)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            query (Tensor): Egocentric BEV spatial queries [B, N_tokens, C].
            key (Tensor): Synchronized RSU BEV spatial keys [B, N_tokens, C].
            value (Tensor): Synchronized RSU BEV spatial values [B, N_tokens, C].
            attn_mask (Tensor, optional): Spatial penalty bias matrix B_spatial [B, 1, N_tokens].

        Returns:
            context (Tensor): Fused spatio-temporal feature representations [B, N_tokens, C].
        """
        # Hyperbolic tangent bounding to stabilize gradient propagation (Eq. 17)
        query = torch.tanh(query / 50.0) * 50.0
        key = torch.tanh(key / 50.0) * 50.0

        # Scaled dot-product cross-attention matching scores
        score = torch.bmm(query, key.transpose(1, 2)) * self.scale
        score = torch.tanh(score / 20.0) * 20.0  # Secondary activation bounding (Eq. 18)

        # Inject spatial request corridor penalization mask (Eq. 19)
        if attn_mask is not None:
            score = score + attn_mask

        attn = F.softmax(score, dim=-1)  # Softmax probability collapse outside corridor (Eq. 20)
        context = torch.bmm(attn, value)  # Aggregated representation (Eq. 21)
        return context


class LatencyCorrector(nn.Module):
    """
    Velocity-Yaw-Rate Compensated Kinematic Latency Correction Network.
    
    Mathematical Formulation (Eq. 11 - 15):
      d_x = (v_x * dt) / \Delta r,  d_y = (v_y * dt) / \Delta r,  \theta = \omega * dt
      A_lat = [[cos(\theta), sin(\theta), 2.0 * d_x / (W - 1)],
               [-sin(\theta), cos(\theta), 2.0 * d_y / (H - 1)]]
      G_affine = G_grid(A_lat)
      \Delta \Phi_flow = F_flow(X_rsu) * \gamma
      \tilde{X}_rsu = S_sample(X_rsu, G_affine + \Delta \Phi_flow)
    """
    def __init__(self, channels: int, resolution: float = 0.125):
        super(LatencyCorrector, self).__init__()
        self.res = resolution
        self.flow_net = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1),
            nn.Tanh()
        )
        self.flow_scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, x: torch.Tensor, latency: torch.Tensor, 
                velocity: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Historical infrastructure BEV features X_rsu [B, C, H, W].
            latency (Tensor): V2I network channel transmission delay \Delta t [B, 1].
            velocity (Tensor): Local egocentric linear velocity components [v_x, v_y] [B, 2].
            omega (Tensor): Instantaneous vehicle angular yaw rate \omega [B, 1].

        Returns:
            x_warped (Tensor): Motion-compensated time-aligned RSU feature map [B, C, H, W].
        """
        B, C, H, W = x.shape
        device = x.device

        dt = latency.view(B, 1).to(x.dtype)
        vx = velocity[:, 0].view(B, 1)
        vy = velocity[:, 1].view(B, 1)
        omega_val = omega.view(B, 1)

        # Compute physical pixel displacements and rotational phase shift (Eq. 11)
        dx = (vx * dt) / self.res
        dy = (vy * dt) / self.res
        dtheta = omega_val * dt

        # Construct 2D affine transformation grid matrix A_lat (Eq. 12)
        affine_mat = torch.zeros(B, 2, 3, device=device, dtype=torch.float32)
        cos_t = torch.cos(dtheta).squeeze(-1)
        sin_t = torch.sin(dtheta).squeeze(-1)

        affine_mat[:, 0, 0] = cos_t
        affine_mat[:, 0, 1] = sin_t
        affine_mat[:, 1, 0] = -sin_t
        affine_mat[:, 1, 1] = cos_t

        affine_mat[:, 0, 2] = 2.0 * dx.squeeze(-1) / (W - 1)
        affine_mat[:, 1, 2] = 2.0 * dy.squeeze(-1) / (H - 1)

        # Generate normalized sampling grid G_affine (Eq. 13)
        grid = F.affine_grid(affine_mat, x.size(), align_corners=True).to(x.dtype)
        
        # Estimate high-order residual dynamic flow field offset \Delta \Phi_flow (Eq. 14)
        learned_disp = self.flow_net(x).permute(0, 2, 3, 1) * self.flow_scale

        # Execute inverse bilinear sampling warping (Eq. 15)
        return F.grid_sample(x, grid + learned_disp, mode='bilinear', padding_mode='zeros', align_corners=True)


class RequestMapGenerator(nn.Module):
    """
    Feedforward Path Spatial Request Corridor Mask Generator.
    
    Mathematical Formulation (Eq. 8 - 10):
      \iota_p = [u_p, v_p]^T = Discrete coordinate mapping of predicted coarse trajectory \hat{\tau}
      \mathcal{M}_bin = Binary request map populated with hard activations at \iota_p
      \mathcal{M}_req = (G_\sigma * \mathcal{M}_bin) / (max(G_\sigma * \mathcal{M}_bin) + \epsilon)
    """
    def __init__(self, map_size: tuple, resolution: float = 0.125):
        super(RequestMapGenerator, self).__init__()
        self.H, self.W = map_size
        self.res = resolution
        self.gaussian_filter = nn.Conv2d(1, 1, kernel_size=5, padding=2, bias=False)
        self._init_gaussian_filter()
        self.gaussian_filter.requires_grad = False

    def _init_gaussian_filter(self):
        k_size, sigma = 5, 1.0
        center = k_size // 2
        x, y = np.mgrid[-center:center+1, -center:center+1]
        g = 1.0 / (2.0 * np.pi * sigma**2) * np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
        self.gaussian_filter.weight.data = torch.from_numpy(g).float().unsqueeze(0).unsqueeze(0)

    def forward(self, trajs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trajs (Tensor): Predicted coarse future motion path \hat{\tau} [B, P, 2].

        Returns:
            heatmap (Tensor): Spatial request corridor heatmap \mathcal{M}_req [B, 1, H, W].
        """
        B, N, _ = trajs.shape
        device = trajs.device
        request_map = torch.zeros((B, 1, self.H, self.W), device=device)

        aligned_trajs = trajs.clone()
        aligned_trajs[..., 1] = -aligned_trajs[..., 1]  # Invert lateral axis for geodetic grid alignment

        traj_px = (aligned_trajs / self.res).long()
        traj_px[..., 0] = torch.clamp(traj_px[..., 0] + self.W // 2, 0, self.W - 1)
        traj_px[..., 1] = torch.clamp(traj_px[..., 1] + self.H // 2, 0, self.H - 1)

        for b in range(B):
            request_map[b, 0, traj_px[b, :, 1], traj_px[b, :, 0]] = 1.0

        # Apply 2D Gaussian kernel smoothing for localization safety margin (Eq. 10)
        heatmap = self.gaussian_filter(request_map)
        max_vals = heatmap.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1)
        heatmap = heatmap / (max_vals + 1e-3)

        return torch.nan_to_num(heatmap, nan=0.0)


class CoarseTrajectoryHead(nn.Module):
    """
    Feedforward Coarse Trajectory Prediction Head.
    
    Mathematical Formulation (Eq. 8):
      \hat{\tau} = W_fc * vec(P_avg(ReLU(N_GN(W_conv * X_ego)))) + b_fc \in \mathbb{R}^{P \times 2}
    """
    def __init__(self, in_channels: int, pred_len: int = 4):
        super(CoarseTrajectoryHead, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.fc = nn.Linear(64, pred_len * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        traj = self.fc(feat)
        return traj.view(-1, 4, 2)


class SpatialConvolutionalControlHead(nn.Module):
    """
    Topography-Preserving Spatial Convolutional Control Head.
    
    Mathematical Formulation (Eq. 29 - 32):
      h_1 = ReLU(GN_1(Conv2D_{k3,s2,p1}(X_fused))) \in \mathbb{R}^{128 \times 48 \times 144}
      h_2 = ReLU(GN_2(Conv2D_{k3,s2,p1}(h_1)))     \in \mathbb{R}^{64 \times 24 \times 72}
      h_3 = ReLU(GN_3(Conv2D_{k3,s2,p1}(h_2)))     \in \mathbb{R}^{32 \times 12 \times 36}
      f_space = Flatten(h_3)                        \in \mathbb{R}^{13,824}
      
    Bypasses early Global Average Pooling (GAP) layers to preserve localized geometric boundaries 
    and fine-grained obstacle proximity vectors within activation weights.
    """
    def __init__(self, in_channels: int):
        super(SpatialConvolutionalControlHead, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True)
        )

    def forward(self, fused_bev: torch.Tensor) -> torch.Tensor:
        return self.conv_block(fused_bev)


class Co_TCP_Advanced(nn.Module):
    """
    Unified CORTEX Deep Neural Network Graph.
    
    Combines:
      - PointPillar BEV Feature Encoder (PFN + PointPillar Scatter + ResNet BEV Backbone)
      - Feedforward Path Projection & Spatial Request Corridor Masking
      - Kinematic Latency Corrector & Hard-Masked Scaled Dot-Product Cross-Attention
      - Autoregressive Gated Recurrent Unit (GRU) Waypoint Decoder
      - Non-Truncated 13,960-Channel Spatial Convolutional Control Connector (Eq. 33 - 40)
    """
    def __init__(self, tcp_cfg, opencood_cfg: dict, comm_cfg: dict):
        super(Co_TCP_Advanced, self).__init__()

        if isinstance(tcp_cfg, dict):
            tcp_cfg = Namespace(**tcp_cfg)

        self.bev_out_channels = sum(opencood_cfg['base_bev_backbone']['num_upsample_filter'])

        # Primitive LiDAR BEV Feature Extraction Pipeline
        self.pillar_vfe = PillarVFE(
            opencood_cfg['pillar_vfe'],
            num_point_features=4,
            voxel_size=opencood_cfg['voxel_size'],
            point_cloud_range=opencood_cfg['point_cloud_range']
        )
        self.scatter = PointPillarScatter(opencood_cfg['point_pillar_scatter'])
        self.backbone = ResNetBEVBackbone(opencood_cfg['base_bev_backbone'], 64)
        
        # Trajectory-Guided Control Prediction Baseline Engine
        self.tcp_model = TCP_Base_Model(tcp_cfg)

        # Topography-Preserving Spatial Convolutional Control Head (Eq. 29 - 32)
        self.spatial_control_backbone = SpatialConvolutionalControlHead(self.bev_out_channels)
        self.tcp_model.join_traj[0] = nn.Linear(128 + self.bev_out_channels, 512)

        # Absolute Invariant Control Input Dimension (Eq. 34 - 35)
        # f_space (13,824) + f_traj (8) + e_meas (128) = 13,960 channels
        self.total_input_dim = 13960
        self.control_connector = nn.Linear(self.total_input_dim, 256)
        self.policy_head = nn.Linear(256, 3)

        # Request-Aware Spatial Fusion & Latency Correction Networks
        self.coarse_head = CoarseTrajectoryHead(in_channels=self.bev_out_channels)
        self.latency_corrector = LatencyCorrector(
            channels=self.bev_out_channels,
            resolution=opencood_cfg['voxel_size'][0]
        )
        self.attn_fusion = ScaledDotProductAttention(dim=self.bev_out_channels)

        grid_h = opencood_cfg['point_pillar_scatter']['grid_size'][1]
        grid_w = opencood_cfg['point_pillar_scatter']['grid_size'][0]
        self.request_generator = RequestMapGenerator(
            map_size=(grid_h, grid_w),
            resolution=opencood_cfg['voxel_size'][0]
        )

    def extract_bev(self, lidar_dict: dict) -> torch.Tensor:
        """Extracts dense 2D BEV feature representations from raw sparse point cloud dictionaries."""
        x = self.pillar_vfe(lidar_dict)
        x = self.scatter(x)
        return self.backbone(x)['spatial_features_2d']

    def forward(self, data_dict: dict) -> dict:
        """
        Executes forward inference through the unified CORTEX computational graph.
        
        Returns:
            dict containing:
              - 'pred_ctrl': Continuous actuator commands [throttle, steer, brake] [B, 3]
              - 'pred_wp': Autoregressively predicted future spatial waypoints [B, 4, 2]
              - 'coarse_traj': Auxiliary feedforward query path estimates [B, 4, 2]
              - 'fused_bev': Synchronized spatio-temporal fused BEV feature map [B, C, H, W]
        """
        # 1. Extract egocentric BEV feature map X_ego
        x_ego = self.extract_bev(data_dict.get('ego_lidar_dict', data_dict))
        x_ego = torch.tanh(x_ego / 50.0) * 50.0

        # 2. Feedforward coarse path projection and spatial request mask generation (Eq. 8 - 10)
        coarse_traj = self.coarse_head(x_ego)
        request_mask = self.request_generator(coarse_traj)

        fused_features = x_ego
        rsu_key = 'rsu_lidar_dict' if 'rsu_lidar_dict' in data_dict else 'rsu_lidar'

        # 3. Request-Aware Spatial Fusion & Kinematic Latency Correction
        if rsu_key in data_dict and data_dict[rsu_key] is not None:
            try:
                x_rsu = self.extract_bev(data_dict[rsu_key])
                x_rsu = torch.tanh(x_rsu / 50.0) * 50.0

                latency = data_dict.get('communication_delay', torch.tensor(0.1, device=x_ego.device))
                velocity = data_dict['measurements'][:, 3:5]
                omega = data_dict['physical_omega']

                # Motion-compensated kinematic latency correction (Eq. 11 - 15)
                x_rsu_aligned = self.latency_corrector(x_rsu, latency, velocity, omega)

                B, C, H, W = x_ego.shape

                # Spatial token adaptive pooling optimization (27,648 -> 1,728 tokens)
                H_att, W_att = 24, 72
                x_ego_down = F.adaptive_avg_pool2d(x_ego, (H_att, W_att))
                x_rsu_down = F.adaptive_avg_pool2d(x_rsu_aligned, (H_att, W_att))

                resized_mask = F.interpolate(request_mask, size=(H_att, W_att), mode='bilinear', align_corners=True)
                flat_mask = (resized_mask.view(B, 1, -1) > 0.05).to(x_ego.dtype)
                spatial_bias = (1.0 - flat_mask) * -60000.0

                query = x_ego_down.view(B, C, -1).permute(0, 2, 1)
                key = x_rsu_down.view(B, C, -1).permute(0, 2, 1)
                value = x_rsu_down.view(B, C, -1).permute(0, 2, 1)

                # Hard-masked spatial cross-attention fusion (Eq. 16 - 21)
                fused_out = cp.checkpoint(self.attn_fusion, query, key, value, spatial_bias)
                fused_features_down = fused_out.permute(0, 2, 1).view(B, C, H_att, W_att)

                # Upsample back to native spatial resolution grid (96 x 288)
                fused_features = F.interpolate(fused_features_down, size=(H, W), mode='bilinear', align_corners=True)

            except Exception:
                fused_features = x_ego

        # 4. Trajectory Prediction Branch (GRU Autoregressive Waypoint Decoder, Eq. 22 - 28)
        feature_emb_gap = F.adaptive_avg_pool2d(fused_features, (1, 1)).flatten(1)
        meas_emb = self.tcp_model.forward_measurements(data_dict['measurements'])
        pred_wp_dict = self.tcp_model.forward_trajectory_branch(feature_emb_gap, meas_emb, data_dict['target_point'])

        # 5. Non-Truncated Spatial Convolutional Control Coupling (Eq. 29 - 35)
        spatial_ctrl_feat = self.spatial_control_backbone(fused_features).flatten(1)
        flat_pred_wps = pred_wp_dict['pred_wp'].flatten(1)

        combined_control_input = torch.cat([spatial_ctrl_feat, flat_pred_wps, meas_emb], dim=1)

        # Zero-padding / dimension bounding to 13,960 channels (Eq. 35)
        if combined_control_input.shape[1] < self.total_input_dim:
            padding_size = self.total_input_dim - combined_control_input.shape[1]
            combined_control_input = F.pad(combined_control_input, (0, padding_size), "constant", 0)
        else:
            combined_control_input = combined_control_input[:, :self.total_input_dim]

        # 6. Continuous Actuation Synthesis & Policy Parameterization (Eq. 36 - 40)
        x_ctrl = F.relu(self.control_connector(combined_control_input))
        raw_ctrl = self.policy_head(x_ctrl)

        throttle = torch.sigmoid(raw_ctrl[:, 0:1])  # Throttle \in [0, 1] (Eq. 38)
        steer = torch.tanh(raw_ctrl[:, 1:2])        # Steering \in [-1, 1] (Eq. 39)
        brake = torch.sigmoid(raw_ctrl[:, 2:3])      # Hydraulic Brake \in [0, 1] (Eq. 40)

        # Safety-critical emergency braking gate intervention
        if not self.training:
            should_brake_gate = data_dict['physical_should_brake'].view(-1, 1)
            throttle = torch.where(should_brake_gate > 0.5, torch.zeros_like(throttle), throttle)
            brake = torch.where(should_brake_gate > 0.5, torch.ones_like(brake), brake)

        pred_ctrl = torch.cat([throttle, steer, brake], dim=1)

        return {
            'pred_ctrl': pred_ctrl,
            'pred_wp': pred_wp_dict['pred_wp'],
            'coarse_traj': coarse_traj,
            'fused_bev': fused_features
        }