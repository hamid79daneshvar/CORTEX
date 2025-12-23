# baselines/tcp_reproduced/config.py
# Official Configuration for the TCP Baseline (Reproduced on V2XVerse)
# Contains hyperparameters for the Vision-Only model and PID controller.

import os

class GlobalConfig:
    """
    Configuration class for the Trajectory-guided Control Prediction (TCP) model.
    Matches the settings used in the original NeurIPS 2022 paper.
    """
    
    # =================================================================
    # 1. Model Architecture & Input Dimensions
    # =================================================================
    seq_len = 1             # Input sequence length (single frame)
    pred_len = 4            # Number of future waypoints to predict
    input_resolution = 256  # Input image resolution (256x256)
    scale = 1               # Image scaling factor
    crop = 256              # Image cropping size
    backbone = 'resnet34'   # Visual encoder backbone
    
    # Perception Head Config
    perception = {
        'ext_type': 'lateral', 
        'n_lat': 3, 
        'use_target_point_image': True, 
        'n_channels': 512   # ResNet34 output feature size
    }
    
    # Measurement Encoder Config
    measurements = {
        'n_input': 9,       # speed(1) + target_point(2) + command(6)
        'n_output': 128     # Embedding size
    }

    # =================================================================
    # 2. PID Controller Parameters
    # =================================================================
    # Used for post-processing trajectory into control actions (if enabled)
    
    # Lateral Control (Steering)
    turn_KP = 0.75
    turn_KI = 0.75
    turn_KD = 0.3
    turn_n = 40             # Integral buffer size

    # Longitudinal Control (Throttle/Brake)
    speed_KP = 5.0
    speed_KI = 0.5
    speed_KD = 1.0
    speed_n = 40            # Integral buffer size

    # Control Thresholds
    max_throttle = 0.75     # Maximum throttle value [0, 1]
    brake_speed = 0.4       # Desired speed threshold to trigger braking
    brake_ratio = 1.1       # Ratio (Current/Desired Speed) to trigger braking
    clip_delta = 0.25       # Clip value for speed error

    # Navigation Parameters
    aim_dist = 4.0          # Lookahead distance for aiming (meters)
    angle_thresh = 0.3      # Angle threshold for steering correction
    dist_thresh = 10        # Distance threshold for target switching
    speed_thresh = 0.1      # Low speed threshold

    # =================================================================
    # 3. Training Hyperparameters (Default)
    # =================================================================
    lr = 1e-4               # Learning Rate
    batch_size = 32
    num_workers = 4