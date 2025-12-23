# train.py
# Official Implementation of CORTEX Training Pipeline
# Standardized for public release

import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Environmental variable to handle OpenMP conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Import CORTEX modules
from model_v2i import Co_TCP_Advanced  # The proposed CORTEX model
from data import V2XVerse_TCP_Dataset, tcp_collate_fn
from config import GlobalConfig

class CoTCP_Trainer(pl.LightningModule):
    """
    PyTorch Lightning Module for training the CORTEX framework.
    Handles the training loop, validation loop, and optimization configuration.
    """
    def __init__(self, config, lr):
        super().__init__()
        # Save hyperparameters for checkpoint reproducibility
        self.save_hyperparameters('config', 'lr')
        self.config_obj = config
        
        # Initialize the CORTEX model (Co-TCP Advanced)
        self.model = Co_TCP_Advanced(
            config.tcp_params, 
            config.opencood_params, 
            config.comm_params
        )
        self.learning_rate = lr

    def training_step(self, batch, batch_idx):
        """
        Execute one training step.
        """
        # Skip empty batches if any
        if batch is None or batch['measurements'].shape[0] == 0:
            return None

        # Forward pass
        pred = self.model(batch)

        # Retrieve loss weights from config
        wp_w = self.config_obj.train_params['wp_loss_weight']
        ctrl_w = self.config_obj.train_params['control_loss_weight']
        # Note: speed_loss is deprecated in this version as per architecture design
        
        # Calculate losses
        wp_loss = F.l1_loss(pred['pred_wp'], batch['waypoints'])
        control_loss = F.l1_loss(pred['pred_ctrl'], batch['control_gt'])
        
        # Total loss aggregation
        loss = wp_w * wp_loss + ctrl_w * control_loss
        
        # Logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_wp_loss', wp_loss, on_step=False, on_epoch=True, logger=True)
        self.log('train_ctrl_loss', control_loss, on_step=False, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Execute one validation step.
        """
        if batch is None or batch['measurements'].shape[0] == 0:
            return None

        # Forward pass
        pred = self.model(batch)

        # Retrieve loss weights
        wp_w = self.config_obj.train_params['wp_loss_weight']
        ctrl_w = self.config_obj.train_params['control_loss_weight']

        # Calculate losses
        wp_loss = F.l1_loss(pred['pred_wp'], batch['waypoints'])
        control_loss = F.l1_loss(pred['pred_ctrl'], batch['control_gt'])
        
        # Total validation loss
        val_loss = wp_w * wp_loss + ctrl_w * control_loss

        # Logging
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_wp_loss', wp_loss, on_step=False, on_epoch=True, logger=True)
        self.log('val_ctrl_loss', control_loss, on_step=False, on_epoch=True, logger=True)

        return val_loss

    def configure_optimizers(self):
        """
        Setup the optimizer. Using AdamW as standard.
        """
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-2)
        return optimizer

if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Train CORTEX Model on V2XVerse Dataset")
    
    # Path arguments
    parser.add_argument('--id', type=str, default='cortex_v1', help='Unique experiment identifier')
    parser.add_argument('--logdir', type=str, default='./logs', help='Directory to save logs and checkpoints')
    parser.add_argument('--raw_data_root', type=str, default='./dataset', help='Root path to the V2XVerse dataset')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=40, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    
    # Hardware arguments
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use')

    args = parser.parse_args()

    # Create log directory structure
    log_path = os.path.join(args.logdir, args.id)
    os.makedirs(log_path, exist_ok=True)

    # Initialize Configuration
    config = GlobalConfig()

    # --- Dataset Split Definition ---
    # Training set includes diverse scenarios from Towns 01, 02, 03, 04, and 06
    train_towns = ['town01', 'town02', 'town03', 'town04', 'town06']
    
    # Validation set includes Towns 07 and 10 (Unseen during training logic)
    # Note: Town05 is reserved strictly for TESTING (Evaluation)
    val_towns = ['town07', 'town10']

    print(f"Initializing Dataset...")
    print(f"Data Root: {args.raw_data_root}")
    print(f"Training Towns: {train_towns}")
    print(f"Validation Towns: {val_towns}")

    # Initialize Datasets
    try:
        train_set = V2XVerse_TCP_Dataset(
            raw_data_root=args.raw_data_root, 
            config=config, 
            split='train', 
            town_filter=train_towns
        )
        val_set = V2XVerse_TCP_Dataset(
            raw_data_root=args.raw_data_root, 
            config=config, 
            split='val', 
            town_filter=val_towns
        )
        print("Datasets created successfully.")
    except Exception as e:
        print(f"Error creating datasets: {e}")
        print("Please verify the dataset path and ensure 'dataset_index.txt' exists.")
        exit(1)

    # Initialize Dataloaders
    train_loader = DataLoader(
        train_set, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True, 
        collate_fn=tcp_collate_fn
    )
    
    val_loader = DataLoader(
        val_set, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True, 
        collate_fn=tcp_collate_fn
    )
    
    # --- Callbacks ---
    # Save the best model based on validation loss
    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss", 
        mode="min", 
        save_top_k=2, 
        save_last=True, 
        dirpath=log_path, 
        filename="best_model-{epoch:02d}-{val_loss:.3f}"
    )
    
    # Stop training if validation loss stops improving
    early_stop_cb = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        verbose=True, 
        mode='min'
    )
    
    # --- Trainer Initialization ---
    trainer = pl.Trainer(
        max_epochs=args.epochs, 
        default_root_dir=log_path, 
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus if args.gpus > 0 else 1,
        strategy='ddp_find_unused_parameters_true' if args.gpus > 1 else 'auto',
        callbacks=[checkpoint_cb, early_stop_cb],
        log_every_n_steps=50
    )

    # Initialize Model
    model = CoTCP_Trainer(config, args.lr)

    # Start Training
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)