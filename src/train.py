"""
Training Script for Deepfake Detection.

End-to-end training pipeline with logging, checkpointing, and validation.
"""
import os
import sys
import argparse
import logging
import random
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import Config, get_config, setup_logger
from models import DualStreamDetector, CombinedLoss
from data import DeepfakeDataset, get_train_transforms, get_val_transforms

logger = setup_logger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_model(config: Config) -> DualStreamDetector:
    """Create and initialize model."""
    model = DualStreamDetector(
        rgb_feature_dim=config.model.rgb_feature_dim,
        freq_feature_dim=config.model.freq_feature_dim,
        fusion_dim=config.model.fusion_dim,
        num_classes=config.model.num_classes,
        dropout=config.model.dropout,
        pretrained=config.model.rgb_pretrained,
        dct_block_size=config.model.dct_block_size,
        freq_channels=config.model.freq_channels,
    )
    return model


def create_optimizer(
    model: nn.Module, 
    config: Config
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Create optimizer and scheduler with differential learning rates."""
    param_groups = model.get_trainable_params()
    
    optimizer = AdamW(
        [
            {
                "params": pg["params"],
                "lr": config.training.learning_rate * pg["lr_scale"],
            }
            for pg in param_groups
        ],
        weight_decay=config.training.weight_decay,
        betas=config.training.betas,
    )
    
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.training.epochs // 3,
        T_mult=2,
        eta_min=config.training.learning_rate * 0.01,
    )
    
    return optimizer, scheduler


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: Detection model.
        train_loader: Training data loader.
        criterion: Combined loss function.
        optimizer: Optimizer.
        device: Device (cuda/cpu).
        epoch: Current epoch number.
        
    Returns:
        Dictionary of average metrics.
    """
    model.train()
    
    total_loss = 0.0
    total_ce = 0.0
    total_con = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits, embeddings = model(images, return_embeddings=True)
        
        # Compute loss
        loss, loss_dict = criterion(logits, embeddings, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Metrics
        total_loss += loss_dict["loss_total"]
        total_ce += loss_dict["loss_ce"]
        total_con += loss_dict["loss_contrastive"]
        
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss_dict['loss_total']:.4f}",
            "acc": f"{100 * correct / total:.2f}%",
        })
    
    num_batches = len(train_loader)
    metrics = {
        "train_loss": total_loss / num_batches,
        "train_ce": total_ce / num_batches,
        "train_con": total_con / num_batches,
        "train_acc": 100 * correct / total,
    }
    
    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    Validate model.
    
    Args:
        model: Detection model.
        val_loader: Validation data loader.
        criterion: Loss function.
        device: Device.
        
    Returns:
        Dictionary of validation metrics.
    """
    model.eval()
    
    total_loss = 0.0
    all_probs = []
    all_labels = []
    correct = 0
    total = 0
    
    for images, labels in tqdm(val_loader, desc="Validating"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        logits, embeddings = model(images, return_embeddings=True)
        loss, loss_dict = criterion(logits, embeddings, labels)
        
        total_loss += loss_dict["loss_total"]
        
        probs = torch.softmax(logits, dim=1)[:, 1]  # Prob of fake
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    # Compute AUC
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(all_labels, all_probs)
    except Exception as e:
        logger.warning(f"Could not compute AUC: {e}")
        auc = 0.0
    
    metrics = {
        "val_loss": total_loss / len(val_loader),
        "val_acc": 100 * correct / total,
        "val_auc": auc * 100,
    }
    
    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    metrics: Dict[str, float],
    path: Path,
) -> None:
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics,
    }
    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved: {path}")


def train(config: Config) -> None:
    """
    Main training loop.
    
    Args:
        config: Training configuration.
    """
    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Training on device: {device}")
    
    # Create model
    model = create_model(config).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss
    criterion = CombinedLoss(
        temperature=config.training.temperature,
        contrastive_weight=config.training.contrastive_weight,
    )
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer(model, config)
    
    # Create dataloaders
    train_dataset = DeepfakeDataset(
        root=str(config.data.ff_root),
        dataset_type="ff++",
        split="train",
        compression=config.data.compression,
        manipulation_types=config.data.manipulation_types,
        transform=get_train_transforms(config.data.image_size),
    )
    
    val_dataset = DeepfakeDataset(
        root=str(config.data.celeb_df_root),
        dataset_type="celeb_df",
        split="val",
        transform=get_val_transforms(config.data.image_size),
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True,
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Training loop
    best_auc = 0.0
    patience_counter = 0
    
    for epoch in range(1, config.training.epochs + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch}/{config.training.epochs}")
        logger.info(f"{'='*50}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        all_metrics = {**train_metrics, **val_metrics}
        logger.info(
            f"Epoch {epoch}: "
            f"Train Loss={train_metrics['train_loss']:.4f}, "
            f"Train Acc={train_metrics['train_acc']:.2f}%, "
            f"Val Loss={val_metrics['val_loss']:.4f}, "
            f"Val Acc={val_metrics['val_acc']:.2f}%, "
            f"Val AUC={val_metrics['val_auc']:.2f}%"
        )
        
        # Save checkpoint
        if epoch % config.training.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, all_metrics,
                config.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            )
        
        # Save best model
        if val_metrics["val_auc"] > best_auc:
            best_auc = val_metrics["val_auc"]
            save_checkpoint(
                model, optimizer, scheduler, epoch, all_metrics,
                config.checkpoint_dir / "best_model.pt"
            )
            logger.info(f"New best AUC: {best_auc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.training.early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    logger.info(f"\nTraining complete. Best AUC: {best_auc:.2f}%")


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Train Deepfake Detector")
    parser.add_argument(
        "--ff-root",
        type=str,
        default="data/FaceForensics++",
        help="Path to FaceForensics++ dataset",
    )
    parser.add_argument(
        "--celeb-df-root",
        type=str,
        default="data/Celeb-DF-v2",
        help="Path to Celeb-DF dataset",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    
    args = parser.parse_args()
    
    config = get_config()
    config.data.ff_root = Path(args.ff_root)
    config.data.celeb_df_root = Path(args.celeb_df_root)
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.device = args.device
    
    train(config)


if __name__ == "__main__":
    main()
