"""
Deepfake Detection - Colab Training Notebook
=============================================
Integrates the existing src/ pipeline for Google Colab training.

Usage:
    1. Upload this file and the src/ directory to Colab
    2. Run: !pip install -r requirements.txt
    3. Execute cells to train the model

This notebook uses the existing modular codebase:
- src/models: DualStreamDetector, DCTExtractor, losses
- src/config: Configuration dataclasses
- src/data: Dataset loaders
"""

# =============================================================================
# Cell 1: Setup and Imports
# =============================================================================

# %pip install torch torchvision tqdm scikit-learn pillow opencv-python-headless

import os
import sys
import logging
from pathlib import Path

# Find src directory - check multiple possible locations
POSSIBLE_SRC_PATHS = [
    Path('./src'),                                        # Current directory
    Path('../src'),                                       # Parent directory
    Path('/content/src'),                                 # Colab root
    Path('/content/Deepfake-Detection-Research/src'),    # Full repo clone
    Path('/content/deepfake/src'),                       # Alternative name
    Path(__file__).parent / 'src' if '__file__' in dir() else None,  # Relative to script
    Path(__file__).parent.parent / 'src' if '__file__' in dir() else None,
]

SRC_PATH = None
for path in POSSIBLE_SRC_PATHS:
    if path and path.exists() and (path / 'models').exists():
        SRC_PATH = path.resolve()
        break

if SRC_PATH is None:
    print("ERROR: src/ directory not found. Searched locations:")
    for p in POSSIBLE_SRC_PATHS:
        if p:
            print(f"  - {p} (exists: {p.exists()})")
    print("\nPlease ensure src/ directory is uploaded with models/, config.py, etc.")
    sys.exit(1)

print(f"Found src/ at: {SRC_PATH}")
sys.path.insert(0, str(SRC_PATH))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm.auto import tqdm
import numpy as np
from PIL import Image

# Import from existing codebase
from config import Config, get_config, setup_logger, ModelConfig, TrainingConfig, DataConfig
from models import DualStreamDetector, EnhancedDualStreamDetector, CombinedLoss, EnhancedCombinedLoss

logger = setup_logger('colab_training')

# Check device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")

if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# =============================================================================
# Cell 2: Configuration
# =============================================================================

class ColabConfig:
    """Configuration optimized for Colab environment."""
    
    # Paths (adjust for your Colab setup)
    DATA_ROOT = Path('/content/data')
    CHECKPOINT_DIR = Path('/content/checkpoints')
    
    # Model
    USE_ENHANCED_MODEL = True  # Use EnhancedDualStreamDetector with CBAM
    RGB_FEATURE_DIM = 1792
    FREQ_FEATURE_DIM = 256
    FUSION_DIM = 512
    DROPOUT = 0.3
    
    # Training
    EPOCHS = 50
    BATCH_SIZE = 32  # Reduce if OOM
    LEARNING_RATE = 1e-4
    BACKBONE_LR_MULT = 0.1
    WEIGHT_DECAY = 1e-5
    
    # Contrastive
    TEMPERATURE = 0.07
    CONTRASTIVE_WEIGHT = 0.5
    
    # Augmentation
    IMAGE_SIZE = 224
    
    # Scheduler
    SCHEDULER_T0 = 10
    SCHEDULER_ETA_MIN = 1e-6
    
    # Early stopping
    PATIENCE = 10
    
    @classmethod
    def setup_dirs(cls):
        cls.DATA_ROOT.mkdir(parents=True, exist_ok=True)
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


ColabConfig.setup_dirs()


# =============================================================================
# Cell 3: Dataset
# =============================================================================

class ColabDeepfakeDataset(Dataset):
    """
    Flexible dataset for deepfake detection.
    
    Expected structure:
    data_root/
    ├── train/
    │   ├── real/
    │   │   └── *.jpg/png
    │   └── fake/
    │       └── *.jpg/png
    └── val/
        ├── real/
        └── fake/
    """
    
    def __init__(
        self, 
        root: Path, 
        split: str = 'train',
        image_size: int = 224,
        augment: bool = True
    ):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.augment = augment and (split == 'train')
        
        # ImageNet normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        # Load samples
        self.samples = self._load_samples()
        logger.info(f"Loaded {len(self.samples)} samples for {split}")
    
    def _load_samples(self):
        samples = []
        split_dir = self.root / self.split
        
        # Real (label=0)
        real_dir = split_dir / 'real'
        if real_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
                for p in real_dir.glob(ext):
                    samples.append((p, 0))
        
        # Fake (label=1)
        fake_dir = split_dir / 'fake'
        if fake_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
                for p in fake_dir.glob(ext):
                    samples.append((p, 1))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        try:
            img = Image.open(path).convert('RGB')
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            
            img_array = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1))
            
            # Augmentation
            if self.augment:
                # Horizontal flip (50%)
                if torch.rand(1).item() < 0.5:
                    img_tensor = torch.flip(img_tensor, dims=[-1])
                
                # Brightness jitter (±20%)
                if torch.rand(1).item() < 0.5:
                    factor = 1.0 + (torch.rand(1).item() - 0.5) * 0.4
                    img_tensor = torch.clamp(img_tensor * factor, 0, 1)
                
                # Gaussian noise (20%)
                if torch.rand(1).item() < 0.2:
                    noise = torch.randn_like(img_tensor) * 0.01
                    img_tensor = torch.clamp(img_tensor + noise, 0, 1)
            
            # Normalize
            img_tensor = (img_tensor - self.mean) / self.std
            
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
            img_tensor = torch.zeros(3, self.image_size, self.image_size)
        
        return img_tensor, label


# =============================================================================
# Cell 4: Model Creation
# =============================================================================

def create_model(enhanced: bool = True) -> nn.Module:
    """Create deepfake detection model."""
    
    if enhanced:
        model = EnhancedDualStreamDetector(
            rgb_feature_dim=ColabConfig.RGB_FEATURE_DIM,
            freq_feature_dim=ColabConfig.FREQ_FEATURE_DIM,
            fusion_dim=ColabConfig.FUSION_DIM,
            num_classes=2,
            dropout=ColabConfig.DROPOUT,
            pretrained=True,
            use_cbam=True,
            use_srm=True,
            use_multiscale=True,
        )
        logger.info("Created EnhancedDualStreamDetector with CBAM, SRM, and multi-scale DCT")
    else:
        model = DualStreamDetector(
            rgb_feature_dim=ColabConfig.RGB_FEATURE_DIM,
            freq_feature_dim=ColabConfig.FREQ_FEATURE_DIM,
            fusion_dim=ColabConfig.FUSION_DIM,
            num_classes=2,
            dropout=ColabConfig.DROPOUT,
            pretrained=True,
        )
        logger.info("Created DualStreamDetector (baseline)")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    return model


# =============================================================================
# Cell 5: Training Loop
# =============================================================================

class ColabTrainer:
    """Training orchestrator for Colab."""
    
    def __init__(self, model: nn.Module, enhanced_loss: bool = True):
        self.model = model.to(DEVICE)
        self.device = DEVICE
        
        # Loss
        if enhanced_loss:
            self.criterion = EnhancedCombinedLoss(
                temperature=ColabConfig.TEMPERATURE,
                contrastive_weight=ColabConfig.CONTRASTIVE_WEIGHT,
                focal_weight=0.3,
                label_smoothing=0.1,
                use_focal=True,
            )
        else:
            self.criterion = CombinedLoss(
                temperature=ColabConfig.TEMPERATURE,
                contrastive_weight=ColabConfig.CONTRASTIVE_WEIGHT,
            )
        
        # Optimizer with differential LR
        param_groups = model.get_trainable_params()
        self.optimizer = AdamW(
            [
                {
                    'params': pg['params'],
                    'lr': ColabConfig.LEARNING_RATE * pg['lr_scale']
                }
                for pg in param_groups
            ],
            weight_decay=ColabConfig.WEIGHT_DECAY,
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=ColabConfig.SCHEDULER_T0,
            eta_min=ColabConfig.SCHEDULER_ETA_MIN,
        )
        
        # Tracking
        self.best_auc = 0.0
        self.patience_counter = 0
        self.history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_acc': []}
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> dict:
        self.model.train()
        total_loss = 0
        total_ce = 0
        total_con = 0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            logits, embeddings = self.model(images, return_embeddings=True)
            loss, loss_dict = self.criterion(logits, embeddings, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss_dict['loss_total']
            total_ce += loss_dict['loss_ce']
            total_con += loss_dict['loss_contrastive']
            
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f"{loss_dict['loss_total']:.4f}",
                'acc': f"{100 * correct / total:.1f}%"
            })
        
        n = len(dataloader)
        return {
            'loss': total_loss / n,
            'ce': total_ce / n,
            'con': total_con / n,
            'acc': 100 * correct / total
        }
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> dict:
        self.model.eval()
        
        all_probs = []
        all_labels = []
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in tqdm(dataloader, desc="Validating"):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            logits, embeddings = self.model(images, return_embeddings=True)
            loss, loss_dict = self.criterion(logits, embeddings, labels)
            
            total_loss += loss_dict['loss_total']
            
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        # AUC
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(all_labels, all_probs)
        except Exception:
            auc = 0.0
        
        return {
            'loss': total_loss / len(dataloader),
            'acc': 100 * correct / total,
            'auc': auc * 100
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_auc': self.best_auc,
            'history': self.history,
        }
        
        path = ColabConfig.CHECKPOINT_DIR / 'latest.pth'
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = ColabConfig.CHECKPOINT_DIR / 'best.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with AUC: {self.best_auc:.2f}%")
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_auc = checkpoint['best_auc']
        self.history = checkpoint['history']
        return checkpoint['epoch']
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> dict:
        logger.info("=" * 60)
        logger.info("Starting training")
        logger.info(f"Device: {self.device}")
        logger.info(f"Epochs: {ColabConfig.EPOCHS}")
        logger.info(f"Batch size: {ColabConfig.BATCH_SIZE}")
        logger.info(f"Learning rate: {ColabConfig.LEARNING_RATE}")
        logger.info("=" * 60)
        
        for epoch in range(1, ColabConfig.EPOCHS + 1):
            logger.info(f"\nEpoch {epoch}/{ColabConfig.EPOCHS}")
            
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader)
            
            self.scheduler.step()
            
            logger.info(
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val AUC: {val_metrics['auc']:.2f}% | "
                f"Val Acc: {val_metrics['acc']:.2f}%"
            )
            
            # Track history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_auc'].append(val_metrics['auc'])
            self.history['val_acc'].append(val_metrics['acc'])
            
            # Check improvement
            is_best = val_metrics['auc'] > self.best_auc
            if is_best:
                self.best_auc = val_metrics['auc']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(epoch, is_best)
            
            if self.patience_counter >= ColabConfig.PATIENCE:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        logger.info(f"\nTraining complete. Best AUC: {self.best_auc:.2f}%")
        return self.history


# =============================================================================
# Cell 6: Visualization
# =============================================================================

def plot_training_history(history: dict):
    """Plot training curves."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Loss
        axes[0].plot(history['train_loss'], label='Train')
        axes[0].plot(history['val_loss'], label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # AUC
        axes[1].plot(history['val_auc'], label='Val AUC', color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('AUC (%)')
        axes[1].set_title('Validation AUC')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Accuracy
        axes[2].plot(history['val_acc'], label='Val Acc', color='orange')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Accuracy (%)')
        axes[2].set_title('Validation Accuracy')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(ColabConfig.CHECKPOINT_DIR / 'training_curves.png', dpi=150)
        plt.show()
        
    except ImportError:
        logger.warning("matplotlib not available for plotting")


# =============================================================================
# Cell 7: Main Execution
# =============================================================================

def main(demo: bool = False):
    """Main training function."""
    
    if demo:
        logger.info("=" * 60)
        logger.info("DEMO MODE - Testing forward pass")
        logger.info("=" * 60)
        
        model = create_model(enhanced=ColabConfig.USE_ENHANCED_MODEL)
        model = model.to(DEVICE)
        model.eval()
        
        dummy_input = torch.randn(4, 3, 224, 224).to(DEVICE)
        
        with torch.no_grad():
            logits, embeddings = model(dummy_input, return_embeddings=True)
        
        logger.info(f"Input shape: {dummy_input.shape}")
        logger.info(f"Logits shape: {logits.shape}")
        logger.info(f"Embeddings shape: {embeddings.shape}")
        logger.info("\n✓ Forward pass successful!")
        return
    
    # Create datasets
    train_dataset = ColabDeepfakeDataset(
        ColabConfig.DATA_ROOT,
        split='train',
        image_size=ColabConfig.IMAGE_SIZE,
        augment=True
    )
    
    val_dataset = ColabDeepfakeDataset(
        ColabConfig.DATA_ROOT,
        split='val',
        image_size=ColabConfig.IMAGE_SIZE,
        augment=False
    )
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        logger.error(
            f"No data found at {ColabConfig.DATA_ROOT}. "
            "Please create train/real, train/fake, val/real, val/fake directories."
        )
        return
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=ColabConfig.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=ColabConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Create model
    model = create_model(enhanced=ColabConfig.USE_ENHANCED_MODEL)
    
    # Create trainer
    trainer = ColabTrainer(model, enhanced_loss=ColabConfig.USE_ENHANCED_MODEL)
    
    # Train
    history = trainer.fit(train_loader, val_loader)
    
    # Plot results
    plot_training_history(history)
    
    logger.info("\nTraining complete!")
    logger.info(f"Best model saved to: {ColabConfig.CHECKPOINT_DIR / 'best.pth'}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='Run demo mode')
    parser.add_argument('--data_root', type=str, default='/content/data')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--enhanced', action='store_true', default=True)
    
    args = parser.parse_args()
    
    ColabConfig.DATA_ROOT = Path(args.data_root)
    ColabConfig.EPOCHS = args.epochs
    ColabConfig.BATCH_SIZE = args.batch_size
    ColabConfig.USE_ENHANCED_MODEL = args.enhanced
    ColabConfig.setup_dirs()
    
    main(demo=args.demo)
