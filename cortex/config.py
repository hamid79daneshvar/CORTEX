# config.py
# Official Configuration File for CORTEX Framework
# Centralizes parameters for TCP Control, OpenCOOD Perception, V2I Communication, and Training.

class GlobalConfig:
    """
    Global configuration object containing all hyperparameters.
    Instantiated in train.py and passed to the model and datasets.
    """
    def __init__(self):
        # =================================================================
        # 1. TCP Baseline Parameters (Planning & Control)
        # =================================================================
        # Parameters inherited from the standard TCP architecture (NeurIPS 2022)
        self.tcp_params = {
            'backbone': 'resnet34',      # Feature extractor for image-based branches (if used)
            'seq_len': 1,                # Input sequence length
            'pred_len': 4,               # Number of future waypoints to predict
            'n_views': 1,                # Number of camera views (1 = Front)
            'n_channels': 256,           # Feature dimension size
            'imagenet_pretrained': True, # Use pre-trained weights for ResNet
            
            # PID Controller Tuning (Used for converting trajectory to control)
            # Longitudinal (Throttle/Brake)
            'speed_KP': 5.0,
            'speed_KI': 0.5,
            'speed_KD': 1.0,
            'speed_n': 40,               # Integral window size
            
            # Lateral (Steering)
            'turn_KP': 1.25,
            'turn_KI': 0.75,
            'turn_KD': 0.3,
            'turn_n': 40,                # Integral window size
            
            # Control Constraints
            'max_throttle': 0.75,        # Cap throttle to prevent erratic acceleration
            'brake_speed': 0.4,          # Speed threshold to trigger braking
            'brake_ratio': 1.1,          # Ratio threshold for braking logic
            'clip_delta': 0.25           # Clipping value for speed error
        }

        # =================================================================
        # 2. OpenCOOD / LiDAR Perception Parameters
        # =================================================================
        # Configuration for PointPillars and Voxelization (LiDAR processing)
        self.opencood_params = {
            # Voxelization Specs (X, Y, Z)
            'voxel_size': [0.4, 0.4, 4], 
            'lidar_range': [-32, -32, -3, 32, 32, 1], # Meters (Left, Back, Down, Right, Front, Up)
            
            # PointPillars Limits
            'max_points_per_voxel': 32,
            'max_voxel_train': 32000,    # Max voxels during training (memory optimization)
            'max_voxel_test': 70000,     # Max voxels during testing (higher fidelity)
            
            # Pillar Feature Net (PFN) Configuration
            'pfn': {
                'use_norm': True,
                'with_distance': False,
                'use_absolute_xyz': True,
                'num_filters': [64],     # PFN output channels
                'num_point_features': 4, # x, y, z, intensity
            },
            
            # Pseudo-Image scatter configuration
            'point_pillar_scatter': {
                'num_features': 64,
                'grid_size': [160, 160, 1], # Resulting BEV map size (Range / Voxel Size)
            },
            
            # BEV Backbone (ResNet-based)
            'base_bev_backbone': {
                "layer_nums": [3, 5, 5],
                "layer_strides": [2, 2, 2],
                "num_filters": [64, 128, 256],
                "upsample_strides": [1, 2, 4],
                "num_upsample_filter": [128, 128, 128],
                "resnet": True,
            }
        }

        # =================================================================
        # 3. V2I Communication Parameters (Request-Aware Fusion)
        # =================================================================
        self.comm_params = {
            'comm_rate': 0.5,             # Ratio of features selected for transmission
            'gaussian_smooth_std': 2.0,   # Standard deviation for Request Map smoothing
            'compression_ratio': 32       # Factor for feature compression (if applicable)
        }

        # =================================================================
        # 4. Training Parameters
        # =================================================================
        self.train_params = {
            # Loss Balancing Weights
            'wp_loss_weight': 1.0,        # Weight for Waypoint L1 Loss
            'control_loss_weight': 1.0,   # Weight for Control Action L1 Loss
            
            # Optimization
            'weight_decay': 1e-2,         # Regularization
            'gradient_clip_val': 5.0,     # Gradient clipping threshold
            
            # Learning Rate Scheduler
            'lr_decay_step': 10,
            'lr_decay_gamma': 0.5
        }