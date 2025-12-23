# baselines/tcp_reproduced/model.py
# Official Re-implementation of the TCP Baseline Model
# Reference: "Trajectory-guided Control Prediction for End-to-end Autonomous Driving" (NeurIPS 2022)

from collections import deque
import numpy as np
import torch 
from torch import nn
from resnet import resnet34 

class PIDController(object):
    """
    Standard PID Controller. 
    Used during inference if trajectory-following control is preferred over direct control prediction.
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
    Trajectory-guided Control Prediction (TCP) - Vision Only Baseline.
    Input: Front Camera Image + Navigation Measurements.
    Output: Control Actions (Steer, Throttle, Brake) + Future Waypoints.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # PID Controllers (Optional usage)
        self.turn_controller = PIDController(K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD, n=config.turn_n)
        self.speed_controller = PIDController(K_P=config.speed_KP, K_I=config.speed_KI, K_D=config.speed_KD, n=config.speed_n)

        # ---------------------------------------------------------
        # 1. Perception Backbone
        # ---------------------------------------------------------
        # ResNet34 pre-trained on ImageNet
        self.perception = resnet34(pretrained=True)
        
        # ---------------------------------------------------------
        # 2. Measurement Encoding
        # ---------------------------------------------------------
        self.measurements_fc = nn.Sequential(
            nn.Linear(config.measurements['n_input'], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # ---------------------------------------------------------
        # 3. Branch A: Trajectory Prediction
        # ---------------------------------------------------------
        self.join_traj = nn.Sequential(
            nn.Linear(512 + 128, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        self.traj_gru = nn.GRUCell(input_size=2, hidden_size=512)
        self.traj_pred = nn.Linear(512, 2) 

        # ---------------------------------------------------------
        # 4. Branch B: Multi-Task Control Prediction
        # ---------------------------------------------------------
        self.speed_branch = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.join_ctrl = nn.Sequential(
            nn.Linear(512 + 128, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3), # [Throttle, Steer, Brake]
        )

        # Auxiliary branches (Legacy from RL distillation)
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

    def forward(self, img, measurements, target_point):
        """
        Forward pass for the baseline model.
        Args:
            img: Front camera image tensor (B, 3, H, W)
            measurements: Vector containing speed, command, etc. (B, 9)
            target_point: Local target coordinates (B, 2)
        """
        # 1. Image Feature Extraction
        feature_emb = self.perception(img) # (B, 512)
        
        # 2. Measurement Encoding
        meas_emb = self.measurements_fc(measurements) # (B, 128)
        
        # 3. Concatenate Features
        feature_vector = torch.cat([feature_emb, meas_emb], dim=1)

        # 4. Predict Speed
        pred_speed = self.speed_branch(feature_emb)

        # 5. Predict Trajectory (Autoregressive GRU)
        output_wp = []
        traj_input = target_point # Initial input
        hidden_state = self.join_traj(feature_vector)

        for _ in range(self.config.pred_len):
            hidden_state = self.traj_gru(traj_input, hidden_state)
            delta_wp = self.traj_pred(hidden_state)
            traj_input = delta_wp 
            output_wp.append(delta_wp)
        
        pred_wp = torch.stack(output_wp, dim=1)

        # 6. Predict Control
        # Fusing features again for control branch
        j_ctrl = self.join_ctrl(feature_vector)
        pred_ctrl_raw = self.policy_head(j_ctrl)
        
        throttle = torch.sigmoid(pred_ctrl_raw[:, 0]).unsqueeze(1)
        steer = torch.tanh(pred_ctrl_raw[:, 1]).unsqueeze(1)
        brake = torch.sigmoid(pred_ctrl_raw[:, 2]).unsqueeze(1)
        
        pred_ctrl = torch.cat([throttle, steer, brake], dim=1)

        return {
            'pred_wp': pred_wp,
            'pred_ctrl': pred_ctrl,
            'pred_speed': pred_speed
        }