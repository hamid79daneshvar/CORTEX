"""
Global Configuration Parameters for the CORTEX Architecture.
Paper: CORTEX: Cooperative Occlusion-Resilient Trajectory Execution via Request-Aware V2I Fusion
Journal: IEEE Access
"""

class GlobalConfig:
    """
    Global configuration class encapsulating model, spatial perception,
    V2I communication, and training hyperparameters for the CORTEX framework.
    """
    def __init__(self):
        # ------------------------------------------------------------------
        # 1. TCP Baseline & Control Parameters (Trajectory-Guided Prediction)
        # ------------------------------------------------------------------
        self.tcp_params = {
            'backbone': 'resnet34',
            'seq_len': 1,
            'pred_len': 4,                  # Look-ahead prediction horizon P=4 steps (0.8s)
            'n_views': 1,
            'n_channels': 256,
            'imagenet_pretrained': True,
            
            # Geometric & Kinematic PID Controller Coefficients
            'turn_KP': 1.25,
            'turn_KI': 0.75,
            'turn_KD': 0.3,
            'turn_n': 40,
            
            'speed_KP': 5.0,
            'speed_KI': 0.5,
            'speed_KD': 1.0,
            'speed_n': 40,
            
            # Actuator Limits & Dynamic Bounds
            'max_throttle': 0.75,
            'brake_speed': 0.4,
            'brake_ratio': 1.1,
            'clip_delta': 0.25
        }

        # ------------------------------------------------------------------
        # 2. Point Cloud Voxelization & BEV Backbone (OpenCOOD / Pillar Encoder)
        # ------------------------------------------------------------------
        self.opencood_params = {
            'voxel_size': [0.125, 0.125, 36.0],        # Spatial cell resolution (0.125m per pixel)
            'point_cloud_range': [-36.0, -12.0, -22.0, 36.0, 12.0, 14.0],  # Spatial ROI: 72m x 24m
            
            # Voxel Density Bounds for VRAM Cap Stabilization
            'max_points_per_voxel': 32,
            'max_voxels': 40000,
            
            'pillar_vfe': {
                'use_norm': True,
                'with_distance': False,
                'use_absolute_xyz': True,
                'num_filters': [64],
                'num_point_features': 4,
            },
            'point_pillar_scatter': {
                'num_features': 64,
                'grid_size': [576, 192, 1],             # 576 * 0.125m = 72m | 192 * 0.125m = 24m
            },
            'base_bev_backbone': {
                'layer_nums': [3, 4, 5],
                'layer_strides': [2, 2, 2],
                'num_filters': [64, 128, 256],
                'upsample_strides': [1, 2, 4],
                'num_upsample_filter': [128, 128, 128], # Concatenated output depth: 384 channels
                'resnet': True,
            }
        }

        # ------------------------------------------------------------------
        # 3. V2I Communication & Latency Compensation Parameters
        # ------------------------------------------------------------------
        self.comm_params = {
            'comm_rate': 0.5,
            'n_head': 8,                                # Multi-head attention count for spatio-temporal fusion
            'gaussian_smooth_std': 2.0,                 # Standard deviation (sigma) for 2D Gaussian query mask
            'latency_bounds': [0.05, 0.30],             # Stochastic V2I transmission lag bounds (50ms - 300ms)
            'resolution': 0.125                         # Spatial BEV cell resolution for affine flow warping
        }

        # ------------------------------------------------------------------
        # 4. Multi-Objective Supervisory Loss & Training Parameters
        # ------------------------------------------------------------------
        self.train_params = {
            'wp_loss_weight': 1.0,                      # alpha: Waypoint trajectory regression weight
            'control_loss_weight': 1.0,                 # beta: Actuator command regression weight
            'coarse_loss_weight': 0.5,                  # gamma: Feedforward coarse trajectory auxiliary weight
            'consistency_loss_weight': 0.1,             # lambda: Kinematic relative displacement consistency weight
            'speed_loss_weight': 0.05,
            'point_dropout_rate': 0.2,                  # Random point cloud dropout augmentation rate
        }