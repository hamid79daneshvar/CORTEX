"""
CORTEX: Cooperative Occlusion-Resilient Trajectory Execution via Request-Aware V2I Fusion
Official Training Pipeline (PyTorch Lightning Implementation)
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

# Cross-platform environment configuration
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step()` before `optimizer.step()`")

from model_v2i import Co_TCP_Advanced
from data import V2XVerse_TCP_Dataset, tcp_collate_fn
from config import GlobalConfig


def replace_bn_with_gn(module: nn.Module) -> None:
    """
    Recursively replaces all BatchNorm layers with GroupNorm layers to stabilize
    gradient updates under small per-GPU batch sizes and mixed-precision training.
    """
    for name, child in module.named_children():
        if isinstance(child, (nn.BatchNorm2d, nn.BatchNorm1d)):
            num_groups = 8 if child.num_features % 8 == 0 else 1
            new_layer = nn.GroupNorm(num_groups=num_groups, num_channels=child.num_features)
            setattr(module, name, new_layer)
        else:
            replace_bn_with_gn(child)


class CoTCP_Trainer(pl.LightningModule):
    """
    PyTorch Lightning Module encapsulating the training, validation, and optimization
    pipeline for the CORTEX framework.
    """

    def __init__(self, config: GlobalConfig, lr: float = 2e-5):
        super().__init__()
        self.save_hyperparameters(ignore=['config'])
        self.config_obj = config
        self.model = Co_TCP_Advanced(
            config.tcp_params,
            config.opencood_params,
            config.comm_params
        )
        replace_bn_with_gn(self.model)
        self.learning_rate = lr

    def training_step(self, batch, batch_idx):
        if batch is None or 'measurements' not in batch or batch['measurements'].shape[0] == 0:
            raise RuntimeError("Invalid or empty batch encountered during training step.")

        pred = self.model(batch)

        # Loss weights according to paper formulation
        wp_w = self.config_obj.train_params.get('wp_loss_weight', 1.0)
        ctrl_w = self.config_obj.train_params.get('control_loss_weight', 1.0)

        gt_waypoints = batch['waypoints_gt']
        gt_control = batch['control_gt']

        # 1. Waypoint trajectory loss (L_wp)
        wp_loss = F.l1_loss(pred['pred_wp'], gt_waypoints) * wp_w

        # 2. Continuous control actuation loss (L_ctrl)
        control_loss = F.l1_loss(pred['pred_ctrl'], gt_control) * ctrl_w

        # 3. Auxiliary request-aware guidance loss (L_coarse)
        coarse_loss = F.l1_loss(pred['coarse_traj'], gt_waypoints) * 0.5

        # 4. Kinematic consistency regularization loss (L_consistency)
        diff_loss = F.mse_loss(pred['pred_wp'][:, 1:], pred['pred_wp'][:, :-1]) * 0.1

        # Unified joint objective function (Eq. 45 in paper)
        total_loss = wp_loss + control_loss + coarse_loss + diff_loss

        if torch.isnan(total_loss):
            raise RuntimeError("Numerical instability detected: total_loss evaluates to NaN.")

        batch_size = batch['measurements'].shape[0]
        self.log_dict({
            'train/total_loss': total_loss,
            'train/wp_loss': wp_loss,
            'train/ctrl_loss': control_loss,
            'train/coarse_loss': coarse_loss,
            'train/consistency': diff_loss
        }, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)

        return total_loss

    def validation_step(self, batch, batch_idx):
        if batch is None or 'measurements' not in batch or batch['measurements'].shape[0] == 0:
            raise RuntimeError("Invalid or empty batch encountered during validation step.")

        pred = self.model(batch)
        gt_waypoints = batch['waypoints_gt']
        gt_control = batch['control_gt']

        wp_loss = F.l1_loss(pred['pred_wp'], gt_waypoints)
        control_loss = F.l1_loss(pred['pred_ctrl'], gt_control)
        val_total_loss = wp_loss + control_loss

        batch_size = batch['measurements'].shape[0]
        self.log('val_loss', val_total_loss, prog_bar=True, on_epoch=True, logger=True, batch_size=batch_size)
        return val_total_loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)

        try:
            total_steps = self.trainer.estimated_stepping_batches
            if total_steps == float('inf') or total_steps <= 0:
                total_steps = 50000
        except Exception:
            total_steps = 50000

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            total_steps=total_steps,
            pct_start=0.2,
            div_factor=10,
            final_div_factor=100
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"}
        }


def main():
    parser = argparse.ArgumentParser(description="CORTEX Training Pipeline")
    parser.add_argument('--id', type=str, default='cortex_v2i_run1', help='Experiment run identifier')
    parser.add_argument('--epochs', type=int, default=40, help='Maximum training epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='Base learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per GPU')
    parser.add_argument('--logdir', type=str, default='training_logs', help='Directory for logs and checkpoints')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to utilize')
    parser.add_argument('--num_workers', type=int, default=4, help='Data loading worker threads')
    parser.add_argument('--raw_data_root', type=str, required=True, help='Root directory of the V2XVerse dataset')

    args = parser.parse_args()
    config = GlobalConfig()

    log_path = Path(args.logdir) / args.id
    raw_data_path = Path(args.raw_data_root)

    trainer_module = CoTCP_Trainer(config, lr=args.lr)

    train_towns = ['town01', 'town02', 'town03', 'town04', 'town06']
    val_towns = ['town07', 'town10']

    print(f"============================================================")
    print(f"CORTEX Training Pipeline Initiated")
    print(f"Training Towns: {train_towns} | Validation Towns: {val_towns}")
    print(f"============================================================")

    train_set = V2XVerse_TCP_Dataset(raw_data_root=raw_data_path, config=config, split='train', town_filter=train_towns)
    val_set = V2XVerse_TCP_Dataset(raw_data_root=raw_data_path, config=config, split='val', town_filter=val_towns)

    is_multiprocessing = args.num_workers > 0

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=tcp_collate_fn,
        persistent_workers=is_multiprocessing
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=tcp_collate_fn,
        persistent_workers=is_multiprocessing
    )

    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        dirpath=str(log_path),
        filename="CORTEX-SOTA-{epoch:02d}-{val_loss:.4f}"
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stop_cb = EarlyStopping(monitor='val_loss', patience=8, mode='min', verbose=True)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        default_root_dir=str(log_path),
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus if args.gpus > 0 else 'auto',
        callbacks=[checkpoint_cb, lr_monitor, early_stop_cb],
        precision=16,
        gradient_clip_val=1.0
    )

    trainer.fit(trainer_module, train_loader, val_loader)


if __name__ == '__main__':
    main()