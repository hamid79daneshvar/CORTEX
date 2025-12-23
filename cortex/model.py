# model.py
# Official Implementation of Trajectory-guided Control Prediction (TCP) Baseline
# Used as the foundation for the trajectory branch in CORTEX.

from collections import deque
import numpy as np
import torch 
from torch import nn
from resnet import resnet34 

class PIDController(object):
    """
    Standard PID Controller implementation for longitudinal and lateral control logic.
    Used during inference to convert waypoints/angles into throttle/brake/steer actions
    if direct control prediction is not used.
    """
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D
        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0

    def step(self, error):
        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = (self._window[-1] - self._window[-2])
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative

class TCP(nn.Module):
    """
    Trajectory-guided Control Prediction (TCP) Model.
    Reference: Wu et al., NeurIPS 2022.
    
    This class implements the vision-only baseline. 
    In CORTEX, we reuse specific sub-modules (like the Trajectory GRU) 
    from this class to maintain architectural consistency.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # PID Controllers for post-processing (optional usage depending on config)
        self.turn_controller = PIDController(K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD, n=config.turn_n)
        self.speed_controller = PIDController(K_P=config.speed_KP, K_I=config.speed_KI, K_D=config.speed_KD, n=config.speed_n)

        # ---------------------------------------------------------
        # 1. Perception Backbone (ResNet34)
        # ---------------------------------------------------------
        # Pre-trained ResNet34 backbone for image feature extraction
        self.perception = resnet34(pretrained=True)
        
        # ---------------------------------------------------------
        # 2. Measurement Encoding
        # ---------------------------------------------------------
        # Encodes high-level commands, speed, and target GPS point
        self.measurements_fc = nn.Sequential(
            nn.Linear(config.measurements['n_input'], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # ---------------------------------------------------------
        # 3. Branch A: Trajectory Prediction (GRU)
        # ---------------------------------------------------------
        # Predicts future waypoints autoregressively
        self.join_traj = nn.Sequential(
            nn.Linear(512 + 128, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        # GRU cell for sequential waypoint generation
        self.traj_gru = nn.GRUCell(input_size=2, hidden_size=512)
        
        # Output layer for waypoints (delta x, delta y)
        self.traj_pred = nn.Linear(512, 2) 

        # ---------------------------------------------------------
        # 4. Branch B: Control Prediction (Multi-Task Heads)
        # ---------------------------------------------------------
        # Speed prediction head
        self.speed_branch = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        # Joint features for control (Perception + Measurements)
        self.join_ctrl = nn.Sequential(
            nn.Linear(512 + 128, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        
        # Control Action Head (Throttle, Steer, Brake)
        self.policy_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3), # Output: [Throttle, Steer, Brake]
        )

        # Value heads (Legacy from RL distillation, kept for compatibility if needed)
        self.value_branch_traj = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.value_branch_ctrl = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    # ---------------------------------------------------------
    # Forward Pass Methods
    # ---------------------------------------------------------

    def forward_measurements(self, measurements):
        """Encode measurement vector."""
        return self.measurements_fc(measurements)

    def traj_branch(self, feature_vector, target_point):
        """
        Autoregressive Trajectory Prediction.
        Generates K future waypoints.
        """
        # Feature vector is concatenation of [Backbone Features, Measurement Embeddings]
        z = self.join_traj(feature_vector)
        
        output_wp = []
        
        # Initial input to GRU is the relative target point
        traj_input = target_point 
        
        # Hidden state initialized with fused features
        hidden_state = z 

        for _ in range(self.config.pred_len):
            hidden_state = self.traj_gru(traj_input, hidden_state)
            delta_wp = self.traj_pred(hidden_state)
            
            # Autoregressive step: next input is the predicted delta
            traj_input = delta_wp 
            
            # Accumulate relative waypoints (simplified relative logic)
            # Note: In CORTEX, post-processing handles the absolute coordinates
            output_wp.append(delta_wp)

        return torch.stack(output_wp, dim=1)

    def control_branch(self, feature_vector, measurements_feature):
        """
        Predicts control actions directly from fused features.
        """
        j_ctrl = self.join_ctrl(torch.cat([feature_vector, measurements_feature], 1))
        pred_ctrl_raw = self.policy_head(j_ctrl)
        
        # Activation functions for control limits
        throttle = torch.sigmoid(pred_ctrl_raw[:, 0]).unsqueeze(1)
        steer = torch.tanh(pred_ctrl_raw[:, 1]).unsqueeze(1)
        brake = torch.sigmoid(pred_ctrl_raw[:, 2]).unsqueeze(1)
        
        pred_ctrl = torch.cat([throttle, steer, brake], dim=1)
        return {'pred_ctrl': pred_ctrl}

    def forward(self, img, measurements, target_point):
        """
        Standard forward pass for TCP (Vision-Only).
        """
        # 1. Perception
        feature_emb = self.perception(img) # (B, 512)
        
        # 2. Measurement Encoding
        meas_emb = self.forward_measurements(measurements)
        
        # 3. Concatenate Features
        # Note: TCP typically concatenates differently based on implementation details.
        # Here we assume a standard concatenation for the branches.
        feature_vector = torch.cat([feature_emb, meas_emb], dim=1)

        # 4. Predictions
        pred_wp = self.traj_branch(feature_vector, target_point)
        pred_ctrl = self.control_branch(feature_emb, meas_emb)
        pred_speed = self.speed_branch(feature_emb)

        return {
            'pred_wp': pred_wp,
            'pred_ctrl': pred_ctrl['pred_ctrl'],
            'pred_speed': pred_speed
        }