"""
model.py
================================================================================
Trajectory-Guided Control Prediction (TCP) Baseline Architecture & PID Controllers

Official PyTorch implementation of the baseline TCP single-agent model.
Configured with an 11-channel kinematic measurement vector:
  [target_x, target_y, normalized_speed, v_x, v_y, command_one_hot (6)]

Target Journal: IEEE Access
Manuscript ID: Access-2025-56941
================================================================================
"""

from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .resnet import resnet34
except ImportError:
    from resnet import resnet34


class PIDController(object):
    """
    Classical Proportional-Integral-Derivative (PID) Controller.
    Used for closed-loop dynamic vehicle lateral and longitudinal control evaluation.
    """
    def __init__(self, K_P: float = 1.0, K_I: float = 0.0, K_D: float = 0.0, n: int = 20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D
        self._window = deque([0.0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0

    def step(self, error: float) -> float:
        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)
        if len(self._window) >= 2:
            integral = float(np.mean(self._window))
            derivative = float(self._window[-1] - self._window[-2])
        else:
            integral = 0.0
            derivative = 0.0
        return self._K_P * error + self._K_I * integral + self._K_D * derivative


class TCP(nn.Module):
    """
    Trajectory-Guided Control Prediction (TCP) Baseline Network Architecture.
    
    Provides measurement encoding, autoregressive GRU trajectory decoding, 
    and policy control prediction heads for single-agent and cooperative baselines.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.turn_controller = PIDController(
            K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD, n=config.turn_n
        )
        self.speed_controller = PIDController(
            K_P=config.speed_KP, K_I=config.speed_KI, K_D=config.speed_KD, n=config.speed_n
        )

        self.perception = resnet34(pretrained=True)

        # 11-channel kinematic measurement encoder MLP (Eq. 1)
        # Vector layout: [target_local (2), norm_speed (1), v_x (1), v_y (1), directional_command_onehot (6)] = 11 scalar channels
        self.measurements = nn.Sequential(
            nn.Linear(11, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )

        self.join_traj = nn.Sequential(
            nn.Linear(128 + 1000, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )

        self.join_ctrl = nn.Sequential(
            nn.Linear(128 + 512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )

        self.speed_branch = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.value_branch_traj = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.value_branch_ctrl = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)
        )

        # Autoregressive recurrent GRU trajectory decoder (Eq. 25 - 28)
        self.decoder_traj = nn.GRUCell(input_size=4, hidden_size=256)
        self.output_traj = nn.Linear(256, 2)

        self.decoder_ctrl = nn.GRUCell(input_size=256 + 4, hidden_size=256)
        self.output_ctrl = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(inplace=True), nn.Linear(256, 256), nn.ReLU(inplace=True)
        )
        self.dist_mu = nn.Sequential(nn.Linear(256, 2), nn.Softplus())
        self.dist_sigma = nn.Sequential(nn.Linear(256, 2), nn.Softplus())
        self.init_att = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(inplace=True), nn.Linear(256, 29 * 8), nn.Softmax(dim=1)
        )
        self.wp_att = nn.Sequential(
            nn.Linear(256 + 256, 256), nn.ReLU(inplace=True), nn.Linear(256, 29 * 8), nn.Softmax(dim=1)
        )
        self.merge = nn.Sequential(
            nn.Linear(512 + 256, 512), nn.ReLU(inplace=True), nn.Linear(512, 256),
        )

    def forward_perception(self, img: torch.Tensor) -> torch.Tensor:
        """Extracts perception feature embeddings from input camera images."""
        feature_emb, _ = self.perception(img)
        return feature_emb

    def forward_measurements(self, state: torch.Tensor) -> torch.Tensor:
        """Projects 11-channel kinematic measurements into 128-dimensional embedding space."""
        return self.measurements(state)

    def forward_trajectory_branch(self, feature_vector: torch.Tensor, 
                                  measurements_feature: torch.Tensor, 
                                  target_point: torch.Tensor) -> dict:
        """
        Autoregressively decodes future waypoints over prediction horizon P=4 (Eq. 25 - 28).
        """
        j_traj = self.join_traj(torch.cat([feature_vector, measurements_feature], dim=1))
        z = j_traj
        output_wp = []

        x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype, device=z.device)

        for _ in range(self.config.pred_len):
            x_in = torch.cat([x, target_point], dim=1)
            z = self.decoder_traj(x_in, z)
            dx = self.output_traj(z)
            x = dx + x
            output_wp.append(x)

        pred_wp = torch.stack(output_wp, dim=1)
        return {'pred_wp': pred_wp}

    def forward_control_branch(self, feature_vector: torch.Tensor, 
                               measurements_feature: torch.Tensor) -> dict:
        """Regresses continuous actuator control commands."""
        j_ctrl = self.join_ctrl(torch.cat([feature_vector, measurements_feature], dim=1))
        pred_ctrl_raw = self.policy_head(j_ctrl)

        throttle = torch.sigmoid(pred_ctrl_raw[:, 0:1])
        steer = torch.tanh(pred_ctrl_raw[:, 1:2])
        brake = torch.sigmoid(pred_ctrl_raw[:, 2:3])

        pred_ctrl = torch.cat([throttle, steer, brake], dim=1)
        return {'pred_ctrl': pred_ctrl}

    def forward(self, image_features: torch.Tensor, measurements_input: torch.Tensor, 
                target_point: torch.Tensor) -> dict:
        """Forward pass for standalone vision TCP model."""
        outputs = {}
        outputs['pred_speed'] = self.speed_branch(image_features)
        measurement_feature = self.forward_measurements(measurements_input)

        traj_output = self.forward_trajectory_branch(image_features, measurement_feature, target_point)
        outputs.update(traj_output)

        ctrl_output = self.forward_control_branch(image_features, measurement_feature)
        outputs.update(ctrl_output)

        return outputs