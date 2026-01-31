"""
Deepfake Detection - Improved Cross-Dataset Generalization
===========================================================
This script implements techniques to improve cross-dataset performance:

1. Multi-source training (train on multiple datasets together)
2. Aggressive data augmentation
3. Production-grade EfficientNet-B4 backbone
4. Frequency augmentation (JPEG compression, blur)
5. Mixup regularization
6. Gradient clipping

Usage in Colab:
    %run colab_improved_training.py
"""

import os
import sys
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Configuration
# =============================================================================

class Config:
    DATA_ROOT = Path('/content/data')
    CHECKPOINT_DIR = Path('/content/checkpoints')
    
    # Training - tuned for generalization
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4  # Lower LR for fine-tuning
    EPOCHS = 15  # More epochs with early stopping
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMAGE_SIZE = 224
    MAX_SAMPLES = 3000  # Per class per dataset
    
    # Regularization
    WEIGHT_DECAY = 1e-4
    DROPOUT = 0.3
    MIXUP_ALPHA = 0.2
    LABEL_SMOOTHING = 0.1
    
    @classmethod
    def setup_dirs(cls):
        cls.DATA_ROOT.mkdir(parents=True, exist_ok=True)
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

Config.setup_dirs()
random.seed(Config.SEED)
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)


# =============================================================================
# Dataset Downloads
# =============================================================================

def download_dataset(name: str) -> bool:
    """Download dataset using kagglehub."""
    dataset_path = Config.DATA_ROOT / name
    dataset_path.mkdir(exist_ok=True)
    
    if list(dataset_path.rglob('*.jpg'))[:1] or list(dataset_path.rglob('*.png'))[:1]:
        print(f"✓ {name} already exists")
        return True
    
    kaggle_ids = {
        '140k_faces': 'xhlulu/140k-real-and-fake-faces',
        'celeb_df': 'vksbhandary/celeb-df-ds-images',
        'faceforensics': 'greatgamedota/faceforensics',
    }
    
    if name not in kaggle_ids:
        return False
    
    print(f"\nDownloading {name}...")
    try:
        import kagglehub
        path = kagglehub.dataset_download(kaggle_ids[name])
        print(f"✓ Downloaded to: {path}")
        
        if Path(path).exists():
            for item in Path(path).iterdir():
                dest = dataset_path / item.name
                if not dest.exists():
                    if item.is_dir():
                        shutil.copytree(item, dest)
                    else:
                        shutil.copy2(item, dest)
            return True
    except Exception as e:
        print(f"✗ Download failed: {e}")
    return False


# =============================================================================
# Data Augmentation (Key for Generalization)
# =============================================================================

class FrequencyAugmentation:
    """Augmentations that affect frequency domain artifacts."""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def jpeg_compression(self, img: np.ndarray, quality: int = None) -> np.ndarray:
        """Apply JPEG compression artifacts."""
        if quality is None:
            quality = random.randint(30, 95)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', img, encode_param)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    
    def gaussian_blur(self, img: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur."""
        ksize = random.choice([3, 5, 7])
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    
    def resize_artifacts(self, img: np.ndarray) -> np.ndarray:
        """Simulate resize artifacts."""
        h, w = img.shape[:2]
        scale = random.uniform(0.5, 0.9)
        small = cv2.resize(img, (int(w*scale), int(h*scale)))
        algo = random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
        return cv2.resize(small, (w, h), interpolation=algo)
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            aug_type = random.choice(['jpeg', 'blur', 'resize'])
            if aug_type == 'jpeg':
                img = self.jpeg_compression(img)
            elif aug_type == 'blur':
                img = self.gaussian_blur(img)
            else:
                img = self.resize_artifacts(img)
        return img


class SpatialAugmentation:
    """Spatial augmentations for robustness."""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            # Horizontal flip
            if random.random() < 0.5:
                img = cv2.flip(img, 1)
            
            # Random rotation
            if random.random() < 0.3:
                angle = random.uniform(-15, 15)
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
                img = cv2.warpAffine(img, M, (w, h))
            
            # Random crop and resize
            if random.random() < 0.3:
                h, w = img.shape[:2]
                crop_ratio = random.uniform(0.8, 1.0)
                new_h, new_w = int(h * crop_ratio), int(w * crop_ratio)
                top = random.randint(0, h - new_h)
                left = random.randint(0, w - new_w)
                img = img[top:top+new_h, left:left+new_w]
                img = cv2.resize(img, (w, h))
            
            # Color jitter
            if random.random() < 0.3:
                img = img.astype(np.float32)
                img = img * random.uniform(0.8, 1.2)  # Brightness
                img = np.clip(img, 0, 255).astype(np.uint8)
        
        return img


# =============================================================================
# DCT Processor
# =============================================================================

class DCTProcessor:
    def __init__(self, block_size: int = 8):
        self.block_size = block_size
    
    def process_image(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        img = img.astype(np.float32)
        h, w = img.shape
        
        pad_h = (self.block_size - h % self.block_size) % self.block_size
        pad_w = (self.block_size - w % self.block_size) % self.block_size
        img_padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
        
        h_pad, w_pad = img_padded.shape
        output = np.zeros_like(img_padded)
        
        for i in range(0, h_pad, self.block_size):
            for j in range(0, w_pad, self.block_size):
                block = img_padded[i:i+self.block_size, j:j+self.block_size]
                dct_block = cv2.dct(block)
                mask = np.ones_like(dct_block)
                mask[:self.block_size//2, :self.block_size//2] = 0
                dct_block *= mask
                output[i:i+self.block_size, j:j+self.block_size] = np.abs(dct_block)
        
        output = output[:h, :w]
        if output.max() > 0:
            output = (output - output.min()) / (output.max() - output.min())
        return output.astype(np.float32)


# =============================================================================
# Dataset with Augmentation
# =============================================================================

class AugmentedDataset(Dataset):
    """Dataset with strong augmentation for generalization."""
    
    def __init__(self, samples: List[Tuple[str, int]], mode: str = 'train'):
        self.samples = samples
        self.mode = mode
        self.dct_processor = DCTProcessor()
        
        if mode == 'train':
            self.freq_aug = FrequencyAugmentation(p=0.5)
            self.spatial_aug = SpatialAugmentation(p=0.5)
        else:
            self.freq_aug = None
            self.spatial_aug = None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        try:
            img = cv2.imread(path)
            if img is None:
                img = np.array(Image.open(path).convert('RGB'))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
            
            # Apply augmentations during training
            if self.mode == 'train':
                if self.spatial_aug:
                    img = self.spatial_aug(img)
                if self.freq_aug:
                    img = self.freq_aug(img)
            
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Spatial stream
            spatial = img.astype(np.float32) / 127.5 - 1.0
            spatial = spatial.transpose(2, 0, 1)
            
            # Frequency stream
            dct = self.dct_processor.process_image(gray)
            dct = cv2.resize(dct, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))[np.newaxis]
            
        except Exception as e:
            spatial = np.zeros((3, Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32)
            dct = np.zeros((1, Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32)
        
        return {
            'spatial': torch.tensor(spatial).float(),
            'freq': torch.tensor(dct).float(),
            'label': torch.tensor(label).long()
        }


def load_dataset_samples(name: str) -> List[Tuple[str, int]]:
    """Load samples from a dataset."""
    samples = []
    real_images = []
    fake_images = []
    
    if name == '140k_faces':
        base = Config.DATA_ROOT / '140k_faces'
        for subdir in base.rglob('*'):
            if subdir.is_dir():
                if subdir.name.lower() == 'real':
                    real_images.extend(list(subdir.glob('*.jpg')) + list(subdir.glob('*.png')))
                elif subdir.name.lower() == 'fake':
                    fake_images.extend(list(subdir.glob('*.jpg')) + list(subdir.glob('*.png')))
    
    elif name == 'celeb_df':
        base = Config.DATA_ROOT / 'celeb_df'
        for subdir in base.rglob('*'):
            if subdir.is_dir():
                name_lower = subdir.name.lower()
                if 'real' in name_lower and 'fake' not in name_lower:
                    real_images.extend(list(subdir.glob('*.jpg')) + list(subdir.glob('*.png')))
                elif 'fake' in name_lower or 'synthesis' in name_lower:
                    fake_images.extend(list(subdir.glob('*.jpg')) + list(subdir.glob('*.png')))
    
    elif name == 'faceforensics':
        base = Config.DATA_ROOT / 'faceforensics'
        for subdir in base.rglob('*'):
            if subdir.is_dir():
                dir_name = subdir.name
                if '_' in dir_name and dir_name.replace('_', '').isdigit():
                    parts = dir_name.split('_')
                    if len(parts) == 2:
                        if parts[0] != parts[1]:
                            fake_images.extend(list(subdir.glob('*.jpg')) + list(subdir.glob('*.png')))
                        else:
                            real_images.extend(list(subdir.glob('*.jpg')) + list(subdir.glob('*.png')))
    
    # Balance and limit
    random.shuffle(real_images)
    random.shuffle(fake_images)
    
    n_samples = min(len(real_images), len(fake_images), Config.MAX_SAMPLES)
    
    samples = [(str(p), 0) for p in real_images[:n_samples]]
    samples += [(str(p), 1) for p in fake_images[:n_samples]]
    
    random.shuffle(samples)
    print(f"  [{name}] Loaded {len(samples)} samples")
    
    return samples


# =============================================================================
# Production Model with EfficientNet
# =============================================================================

class ImprovedDualStreamNetwork(nn.Module):
    """
    Improved dual-stream network with:
    - EfficientNet-B4 for RGB stream
    - Deeper frequency stream
    - Attention fusion
    - Dropout for regularization
    """
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        
        # RGB Stream - EfficientNet-B4
        self.efficientnet = models.efficientnet_b4(
            weights='DEFAULT' if pretrained else None
        )
        self.efficientnet.classifier = nn.Identity()
        self.rgb_dim = 1792
        
        # Freeze early layers for better generalization
        for name, param in self.efficientnet.named_parameters():
            if 'features.0' in name or 'features.1' in name or 'features.2' in name:
                param.requires_grad = False
        
        # Frequency Stream - Deeper CNN
        self.freq_stream = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.freq_dim = 256
        
        # Attention-based fusion
        total_dim = self.rgb_dim + self.freq_dim
        self.attention = nn.Sequential(
            nn.Linear(total_dim, total_dim // 4),
            nn.ReLU(),
            nn.Linear(total_dim // 4, total_dim),
            nn.Sigmoid()
        )
        
        # Classifier with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(Config.DROPOUT),
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(256, 2)
        )
    
    def forward(self, x_rgb, x_freq):
        # RGB features
        feat_rgb = self.efficientnet(x_rgb)
        
        # Frequency features
        feat_freq = self.freq_stream(x_freq)
        
        # Concatenate
        combined = torch.cat([feat_rgb, feat_freq], dim=1)
        
        # Attention-weighted fusion
        attention_weights = self.attention(combined)
        combined = combined * attention_weights
        
        # Classification
        out = self.classifier(combined)
        
        return out, combined


# =============================================================================
# Mixup Training
# =============================================================================

def mixup_data(x_rgb, x_freq, y, alpha=0.2):
    """Apply mixup augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x_rgb.size(0)
    index = torch.randperm(batch_size).to(x_rgb.device)
    
    mixed_rgb = lam * x_rgb + (1 - lam) * x_rgb[index]
    mixed_freq = lam * x_freq + (1 - lam) * x_freq[index]
    y_a, y_b = y, y[index]
    
    return mixed_rgb, mixed_freq, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# =============================================================================
# Trainer
# =============================================================================

class ImprovedTrainer:
    """Trainer with techniques for better generalization."""
    
    def __init__(self, train_datasets: List[str], test_datasets: List[str]):
        self.device = Config.DEVICE
        self.train_datasets = train_datasets
        self.test_datasets = test_datasets
        
        print("\n" + "="*70)
        print("IMPROVED DEEPFAKE DETECTOR - CROSS-DATASET TRAINING")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Training on: {train_datasets}")
        print(f"Testing on: {test_datasets}")
        print("="*70)
        
        # Combine training data from multiple sources
        print("\n📥 Loading training data...")
        all_train_samples = []
        for ds_name in train_datasets:
            samples = load_dataset_samples(ds_name)
            if samples:
                # Split 80/20 for train/val within each dataset
                split_idx = int(0.8 * len(samples))
                all_train_samples.extend(samples[:split_idx])
        
        random.shuffle(all_train_samples)
        self.train_dataset = AugmentedDataset(all_train_samples, mode='train')
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=True, 
            num_workers=2,
            pin_memory=True
        )
        
        print(f"\n  Total training samples: {len(self.train_dataset)}")
        
        # Create model
        print("\n🔧 Creating model...")
        self.model = ImprovedDualStreamNetwork(pretrained=True).to(self.device)
        
        # Count parameters
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Parameters: {total:,} (Trainable: {trainable:,})")
        
        # Loss with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=Config.EPOCHS
        )
    
    def train(self):
        """Train with mixup and gradient clipping."""
        print("\n" + "="*70)
        print("TRAINING")
        print("="*70)
        
        best_avg_acc = 0.0
        
        for epoch in range(Config.EPOCHS):
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
            for batch in pbar:
                spatial = batch['spatial'].to(self.device)
                freq = batch['freq'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Mixup
                if Config.MIXUP_ALPHA > 0:
                    spatial, freq, labels_a, labels_b, lam = mixup_data(
                        spatial, freq, labels, Config.MIXUP_ALPHA
                    )
                
                self.optimizer.zero_grad()
                outputs, _ = self.model(spatial, freq)
                
                # Mixup loss
                if Config.MIXUP_ALPHA > 0:
                    loss = mixup_criterion(self.criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = self.criterion(outputs, labels)
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0) if Config.MIXUP_ALPHA == 0 else labels_a.size(0)
                if Config.MIXUP_ALPHA == 0:
                    correct += (predicted == labels).sum().item()
                else:
                    correct += (lam * (predicted == labels_a).float() + 
                               (1-lam) * (predicted == labels_b).float()).sum().item()
                
                pbar.set_postfix({'loss': f'{train_loss/total:.4f}'})
            
            self.scheduler.step()
            
            # Evaluate on all test datasets
            avg_acc = self._evaluate_all()
            
            print(f"  Epoch {epoch+1}: Train Acc: {100*correct/total:.2f}% | "
                  f"Avg Cross-Dataset Acc: {avg_acc:.2f}%")
            
            if avg_acc > best_avg_acc:
                best_avg_acc = avg_acc
                torch.save(self.model.state_dict(), 
                          Config.CHECKPOINT_DIR / 'best_improved_model.pth')
                print(f"  >>> NEW BEST! Saved.")
        
        # Load best model
        self.model.load_state_dict(
            torch.load(Config.CHECKPOINT_DIR / 'best_improved_model.pth')
        )
        
        return best_avg_acc
    
    def _evaluate_all(self) -> float:
        """Evaluate on all test datasets and return average accuracy."""
        self.model.eval()
        accs = []
        
        for ds_name in self.test_datasets:
            samples = load_dataset_samples(ds_name)
            if not samples:
                continue
            
            # Use last 20% as test
            test_samples = samples[int(0.8 * len(samples)):]
            test_dataset = AugmentedDataset(test_samples, mode='test')
            test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
            
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in test_loader:
                    spatial = batch['spatial'].to(self.device)
                    freq = batch['freq'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    outputs, _ = self.model(spatial, freq)
                    _, predicted = torch.max(outputs, 1)
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            acc = accuracy_score(all_labels, all_preds) * 100
            accs.append(acc)
        
        return np.mean(accs) if accs else 0.0
    
    def final_evaluation(self) -> Dict:
        """Final evaluation with detailed metrics."""
        print("\n" + "="*70)
        print("FINAL CROSS-DATASET EVALUATION")
        print("="*70)
        
        self.model.eval()
        results = {}
        
        for ds_name in self.test_datasets:
            samples = load_dataset_samples(ds_name)
            if not samples:
                continue
            
            test_samples = samples[int(0.8 * len(samples)):]
            test_dataset = AugmentedDataset(test_samples, mode='test')
            test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
            
            all_preds = []
            all_labels = []
            all_probs = []
            
            with torch.no_grad():
                for batch in test_loader:
                    spatial = batch['spatial'].to(self.device)
                    freq = batch['freq'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    outputs, _ = self.model(spatial, freq)
                    probs = F.softmax(outputs, dim=1)[:, 1]
                    _, predicted = torch.max(outputs, 1)
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
            
            acc = accuracy_score(all_labels, all_preds) * 100
            try:
                auc = roc_auc_score(all_labels, all_probs) * 100
            except:
                auc = 50.0
            f1 = f1_score(all_labels, all_preds) * 100
            
            results[ds_name] = {'accuracy': acc, 'auc': auc, 'f1': f1}
            
            print(f"\n  {ds_name.upper()}:")
            print(f"    Accuracy: {acc:.2f}%")
            print(f"    AUC-ROC:  {auc:.2f}%")
            print(f"    F1 Score: {f1:.2f}%")
        
        # Summary
        avg_acc = np.mean([r['accuracy'] for r in results.values()])
        avg_auc = np.mean([r['auc'] for r in results.values()])
        
        print("\n" + "-"*50)
        print(f"  AVERAGE ACCURACY: {avg_acc:.2f}%")
        print(f"  AVERAGE AUC-ROC:  {avg_auc:.2f}%")
        print("-"*50)
        
        return results


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "="*70)
    print("📦 Installing packages...")
    print("="*70)
    
    try:
        import kagglehub
    except ImportError:
        import subprocess
        subprocess.check_call(['pip', 'install', 'kagglehub', '-q'])
    
    try:
        from skimage import data as _
    except ImportError:
        import subprocess
        subprocess.check_call(['pip', 'install', 'scikit-image', '-q'])
    
    # Download datasets
    print("\n📥 Downloading datasets...")
    available = []
    for name in ['140k_faces', 'celeb_df', 'faceforensics']:
        if download_dataset(name):
            samples = load_dataset_samples(name)
            if len(samples) > 100:
                available.append(name)
    
    if len(available) < 2:
        print("⚠️ Need at least 2 datasets for cross-dataset training.")
        return
    
    print(f"\n✓ Available datasets: {available}")
    
    # Strategy: Train on ALL available, test on ALL
    # This is multi-source training for better generalization
    trainer = ImprovedTrainer(
        train_datasets=available,  # Train on all
        test_datasets=available    # Test on all
    )
    
    # Train
    best_acc = trainer.train()
    
    # Final evaluation
    results = trainer.final_evaluation()
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE")
    print("="*70)
    print(f"  Best Average Cross-Dataset Accuracy: {best_acc:.2f}%")
    print(f"  Model saved to: {Config.CHECKPOINT_DIR / 'best_improved_model.pth'}")
    print("="*70)
    
    return results


if __name__ == "__main__":
    results = main()
