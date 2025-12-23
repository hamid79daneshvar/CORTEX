# baselines/tcp_reproduced/train_baseline.py
# Official Re-implementation of TCP Baseline on V2XVerse Dataset
# Ensures fair comparison by training on the same split as CORTEX.

import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Environmental setup
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Import Baseline Modules
# Assumes model.py and data_tcp.py are in the same directory (baselines/tcp_reproduced)
from model import TCP
from data_tcp import V2XVerse_for_TCP_Dataset, tcp_collate_fn

class BaselineConfig:
    """
    Configuration for the TCP Baseline (Vision-Only).
    Matches the settings used in the CORTEX comparison.
    """
    def __init__(self):
        self.pred_len = 4
        self.input_resolution = 256
        self.turn_KP = 0.75; self.turn_KI = 0.75; self.turn_KD = 0.3; self.turn_n = 40
        self.speed_KP = 5.0; self.speed_KI = 0.5; self.speed_KD = 1.0; self.speed_n = 40
        self.img_aug = True 
        self.max_throttle = 0.75; self.brake_speed = 0.4; self.brake_ratio = 1.1; self.clip_delta = 0.25
        self.aim_dist = 4.0; self.angle_thresh = 0.3; self.dist_thresh = 10
        self.speed_thresh = 0.1
        
        # Model specific params
        self.measurements = {'n_input': 9, 'n_output': 128} # speed, target, command
        self.perception = {'n_channels': 512} # ResNet34 output

class TCP_Baseline_Trainer(pl.LightningModule):
    """
    Lightning Module for training the TCP Baseline.
    """
    def __init__(self, config, lr):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.lr = lr
        self.model = TCP(config)

    def forward(self, batch):
        # Forward pass handling data unwrapping
        # TCP model expects: img, measurements, target_point
        return self.model(
            batch['front_img'], 
            torch.cat([batch['speed'], batch['target_point'], batch['target_command']], dim=1),
            batch['target_point']
        )

    def training_step(self, batch, batch_idx):
        if batch is None: return None
        
        # Forward
        pred = self.forward(batch)
        
        # Loss Calculation
        # 1. Trajectory Loss (L1)
        wp_loss = F.l1_loss(pred['pred_wp'], batch['waypoints'])
        
        # 2. Control Loss (L1)
        # Note: We added this head to ensure fair comparison with CORTEX
        control_loss = F.l1_loss(pred['pred_ctrl'], batch['control_gt'])
        
        # 3. Speed Loss (Optional, used in original TCP)
        speed_loss = F.l1_loss(pred['pred_speed'], batch['speed'])

        # Total Loss (Weighted sum)
        loss = wp_loss + control_loss + 0.1 * speed_loss
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_wp_loss', wp_loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if batch is None: return None
        
        pred = self.forward(batch)
        
        wp_loss = F.l1_loss(pred['pred_wp'], batch['waypoints'])
        control_loss = F.l1_loss(pred['pred_ctrl'], batch['control_gt'])
        loss = wp_loss + control_loss

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TCP Baseline on V2XVerse")
    parser.add_argument('--id', type=str, default='tcp_baseline', help='Experiment ID')
    parser.add_argument('--data_root', type=str, default='../../dataset', help='Path to V2XVerse dataset')
    parser.add_argument('--logdir', type=str, default='./logs_baseline', help='Log directory')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpus', type=int, default=1)
    
    args = parser.parse_args()
    
    # Initialize Config
    config = BaselineConfig()
    
    # --- Data Splits (CRITICAL FOR FAIR COMPARISON) ---
    # Must match CORTEX splits. Town05 is EXCLUDED from training.
    train_towns = ['town01', 'town02', 'town03', 'town04', 'town06']
    val_towns = ['town07', 'town10']
    
    print(f"Training Baseline on: {train_towns}")
    print(f"Validating Baseline on: {val_towns}")
    
    # Initialize Datasets
    try:
        train_set = V2XVerse_for_TCP_Dataset(
            root=args.data_root, config=config, split='train', town_filter=train_towns
        )
        val_set = V2XVerse_for_TCP_Dataset(
            root=args.data_root, config=config, split='val', town_filter=val_towns
        )
    except Exception as e:
        print(f"Error loading datasets: {e}")
        exit(1)

    # Dataloaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=tcp_collate_fn, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=tcp_collate_fn, pin_memory=True)
    
    # Trainer
    model = TCP_Baseline_Trainer(config, args.lr)
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss', mode='min', save_top_k=1, save_last=True,
        dirpath=os.path.join(args.logdir, args.id),
        filename='tcp_best-{epoch:02d}-{val_loss:.3f}'
    )
    
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        default_root_dir=os.path.join(args.logdir, args.id),
        accelerator='gpu', devices=args.gpus,
        callbacks=[checkpoint_callback]
    )
    
    print("Starting Baseline Training...")
    trainer.fit(model, train_loader, val_loader)