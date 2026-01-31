"""
Kaggle Deepfake Detection - Fast Parallel Training
===================================================
Optimized for Kaggle notebooks with attached datasets.

Supports:
  - celeb-df-v2
  - faceforensics  
  - caleb-df-v2-preprocessed-to-npz-files

Features:
  - Mixed Precision (2x faster)
  - Multi-GPU DataParallel
  - Fast DataLoader with 4 workers
  - EfficientNet-B4 backbone

Usage in Kaggle:
  1. Attach datasets: celeb-df-v2, faceforensics
  2. Run this cell
"""

import os
import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Kaggle Config
# =============================================================================

class Config:
    # Kaggle paths
    INPUT_DIR = Path('/kaggle/input')
    OUTPUT_DIR = Path('/kaggle/working')
    
    # Datasets (Kaggle format)
    CELEB_DF = INPUT_DIR / 'celeb-df-v2'
    FACEFORENSICS = INPUT_DIR / 'faceforensics'
    CELEB_NPZ = INPUT_DIR / 'caleb-df-v2-preprocessed-to-npz-files'
    
    # Training
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    EPOCHS = 10
    IMAGE_SIZE = 224
    MAX_SAMPLES = 5000
    
    # Parallelism
    NUM_WORKERS = 4
    USE_AMP = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    # Regularization
    WEIGHT_DECAY = 1e-4
    DROPOUT = 0.3
    LABEL_SMOOTHING = 0.1

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# =============================================================================
# Dataset Loaders
# =============================================================================

def find_images(base_path: Path, label: int, keywords: List[str]) -> List[Tuple[str, int]]:
    """Find images in directories matching keywords."""
    samples = []
    if not base_path.exists():
        return samples
    
    for subdir in base_path.rglob('*'):
        if subdir.is_dir():
            name = subdir.name.lower()
            if any(k in name for k in keywords):
                for ext in ['*.jpg', '*.png', '*.jpeg']:
                    samples.extend([(str(p), label) for p in subdir.glob(ext)])
    return samples


def load_celeb_df() -> List[Tuple[str, int]]:
    """Load Celeb-DF v2 dataset."""
    print("Loading Celeb-DF v2...")
    samples = []
    
    base = Config.CELEB_DF
    if not base.exists():
        print(f"  ⚠️ Not found: {base}")
        return samples
    
    # Real faces
    real = find_images(base, 0, ['real', 'youtube', 'original'])
    # Fake faces  
    fake = find_images(base, 1, ['fake', 'synthesis', 'celeb'])
    
    # If structure is different, scan all subdirs
    if len(real) == 0 or len(fake) == 0:
        for subdir in base.iterdir():
            if subdir.is_dir():
                name = subdir.name.lower()
                for ext in ['*.jpg', '*.png', '*.jpeg']:
                    imgs = list(subdir.rglob(ext))
                    if 'real' in name or 'youtube' in name:
                        real.extend([(str(p), 0) for p in imgs])
                    elif 'celeb' in name or 'synth' in name:
                        fake.extend([(str(p), 1) for p in imgs])
    
    n = min(len(real), len(fake), Config.MAX_SAMPLES)
    samples = real[:n] + fake[:n]
    print(f"  ✓ Celeb-DF: {len(samples)} samples (real: {n}, fake: {n})")
    return samples


def load_faceforensics() -> List[Tuple[str, int]]:
    """Load FaceForensics dataset."""
    print("Loading FaceForensics...")
    samples = []
    
    base = Config.FACEFORENSICS
    if not base.exists():
        print(f"  ⚠️ Not found: {base}")
        return samples
    
    all_images = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        all_images.extend(list(base.rglob(ext)))
    
    # FF++ structure: XXX_YYY folders (all manipulated)
    # We'll use directory structure to find real/fake
    real = []
    fake = []
    
    for img in all_images:
        parent = img.parent.name.lower()
        grandparent = img.parent.parent.name.lower() if img.parent.parent != base else ""
        
        if 'original' in parent or 'original' in grandparent or 'real' in parent:
            real.append((str(img), 0))
        elif '_' in parent:  # XXX_YYY format = manipulated
            fake.append((str(img), 1))
        else:
            fake.append((str(img), 1))  # Default to fake for FF++
    
    # If no real found, this dataset is fake-only
    if len(real) == 0:
        print(f"  Note: FF++ has only manipulated images ({len(fake)} fake)")
        return []
    
    n = min(len(real), len(fake), Config.MAX_SAMPLES)
    samples = real[:n] + fake[:n]
    print(f"  ✓ FaceForensics: {len(samples)} samples (real: {n}, fake: {n})")
    return samples


def load_npz_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Load preprocessed NPZ dataset."""
    print("Loading NPZ preprocessed data...")
    
    base = Config.CELEB_NPZ
    if not base.exists():
        print(f"  ⚠️ Not found: {base}")
        return None, None
    
    npz_files = list(base.glob('*.npz'))
    if not npz_files:
        print("  ⚠️ No NPZ files found")
        return None, None
    
    all_images = []
    all_labels = []
    
    for npz_file in npz_files[:5]:  # Limit files
        try:
            data = np.load(npz_file)
            if 'images' in data and 'labels' in data:
                all_images.append(data['images'])
                all_labels.append(data['labels'])
            elif 'X' in data and 'y' in data:
                all_images.append(data['X'])
                all_labels.append(data['y'])
        except Exception as e:
            print(f"  Warning: Failed to load {npz_file}: {e}")
    
    if all_images:
        images = np.concatenate(all_images, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        print(f"  ✓ NPZ: {len(labels)} samples")
        return images, labels
    
    return None, None


# =============================================================================
# DCT Processor
# =============================================================================

class DCTProcessor:
    def __init__(self, block_size: int = 8):
        self.block_size = block_size
    
    def process(self, img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        img = img.astype(np.float32)
        h, w = img.shape
        
        pad_h = (self.block_size - h % self.block_size) % self.block_size
        pad_w = (self.block_size - w % self.block_size) % self.block_size
        img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
        
        h_pad, w_pad = img.shape
        output = np.zeros_like(img)
        
        for i in range(0, h_pad, self.block_size):
            for j in range(0, w_pad, self.block_size):
                block = img[i:i+self.block_size, j:j+self.block_size]
                dct_block = cv2.dct(block)
                mask = np.ones_like(dct_block)
                mask[:4, :4] = 0
                output[i:i+self.block_size, j:j+self.block_size] = np.abs(dct_block * mask)
        
        output = output[:h-pad_h if pad_h else h, :w-pad_w if pad_w else w]
        if output.max() > 0:
            output = output / output.max()
        return output.astype(np.float32)


# =============================================================================
# Dataset
# =============================================================================

class KaggleDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], mode: str = 'train'):
        self.samples = samples
        self.mode = mode
        self.dct = DCTProcessor()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        try:
            img = cv2.imread(path)
            if img is None:
                img = np.array(Image.open(path).convert('RGB'))
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            img = cv2.resize(img, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
            
            # Augmentation for training
            if self.mode == 'train' and random.random() > 0.5:
                img = cv2.flip(img, 1)
            
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Spatial
            spatial = img.astype(np.float32) / 127.5 - 1.0
            spatial = spatial.transpose(2, 0, 1)
            
            # Frequency
            dct = self.dct.process(gray)
            dct = cv2.resize(dct, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))[np.newaxis]
            
        except:
            spatial = np.zeros((3, Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32)
            dct = np.zeros((1, Config.IMAGE_SIZE, Config.IMAGE_SIZE), dtype=np.float32)
        
        return {
            'spatial': torch.tensor(spatial).float(),
            'freq': torch.tensor(dct).float(),
            'label': torch.tensor(label).long()
        }


class NPZDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = images
        self.labels = labels
        self.dct = DCTProcessor()
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        
        if img.shape[0] == 3:  # CHW -> HWC
            img = img.transpose(1, 2, 0)
        
        img = cv2.resize(img, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
        
        spatial = img.astype(np.float32) / 127.5 - 1.0
        if len(spatial.shape) == 2:
            spatial = np.stack([spatial]*3, axis=0)
        else:
            spatial = spatial.transpose(2, 0, 1)
        
        dct = self.dct.process(gray)
        dct = cv2.resize(dct, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))[np.newaxis]
        
        return {
            'spatial': torch.tensor(spatial).float(),
            'freq': torch.tensor(dct).float(),
            'label': torch.tensor(int(label)).long()
        }


# =============================================================================
# Model
# =============================================================================

class DualStreamNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # RGB - EfficientNet-B4
        self.rgb = models.efficientnet_b4(weights='DEFAULT')
        self.rgb.classifier = nn.Identity()
        
        # Freeze early layers
        for name, param in self.rgb.named_parameters():
            if 'features.0' in name or 'features.1' in name:
                param.requires_grad = False
        
        # Frequency stream
        self.freq = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(Config.DROPOUT),
            nn.Linear(1792 + 256, 256),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(256, 2)
        )
    
    def forward(self, x_rgb, x_freq):
        f_rgb = self.rgb(x_rgb)
        f_freq = self.freq(x_freq)
        combined = torch.cat([f_rgb, f_freq], dim=1)
        return self.classifier(combined)


# =============================================================================
# Trainer
# =============================================================================

def train_model(train_loader, val_loader, epochs=Config.EPOCHS):
    print("\n" + "="*60)
    print("🚀 TRAINING")
    print("="*60)
    print(f"Device: {Config.DEVICE}")
    print(f"GPUs: {Config.NUM_GPUS}")
    print(f"Mixed Precision: {Config.USE_AMP}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print("="*60)
    
    model = DualStreamNet().to(Config.DEVICE)
    
    if Config.NUM_GPUS > 1:
        print(f"Using {Config.NUM_GPUS} GPUs")
        model = nn.DataParallel(model)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=Config.USE_AMP)
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            spatial = batch['spatial'].to(Config.DEVICE, non_blocking=True)
            freq = batch['freq'].to(Config.DEVICE, non_blocking=True)
            labels = batch['label'].to(Config.DEVICE, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=Config.USE_AMP):
                outputs = model(spatial, freq)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.1f}%'})
        
        scheduler.step()
        
        # Validation
        val_acc, val_auc = evaluate(model, val_loader)
        print(f"  Val Acc: {val_acc:.2f}% | AUC: {val_auc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), Config.OUTPUT_DIR / 'best_model.pth')
            print("  ✓ Saved best model")
    
    return model, best_acc


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for batch in loader:
            spatial = batch['spatial'].to(Config.DEVICE)
            freq = batch['freq'].to(Config.DEVICE)
            labels = batch['label']
            
            with torch.cuda.amp.autocast(enabled=Config.USE_AMP):
                outputs = model(spatial, freq)
            
            probs = F.softmax(outputs, dim=1)[:, 1]
            _, pred = outputs.max(1)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds) * 100
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5
    
    return acc, auc


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "="*60)
    print("📦 KAGGLE DEEPFAKE DETECTION")
    print("="*60)
    
    # Load all available datasets
    all_samples = []
    
    celeb = load_celeb_df()
    if celeb:
        all_samples.extend(celeb)
    
    ff = load_faceforensics()
    if ff:
        all_samples.extend(ff)
    
    # Try NPZ data
    npz_images, npz_labels = load_npz_dataset()
    
    if not all_samples and npz_images is None:
        print("\n❌ No datasets found!")
        print("Please attach: celeb-df-v2 or faceforensics")
        return
    
    # Create datasets
    if all_samples:
        random.shuffle(all_samples)
        split = int(0.8 * len(all_samples))
        train_samples = all_samples[:split]
        val_samples = all_samples[split:]
        
        train_dataset = KaggleDataset(train_samples, mode='train')
        val_dataset = KaggleDataset(val_samples, mode='val')
    else:
        # Use NPZ
        split = int(0.8 * len(npz_labels))
        idx = np.random.permutation(len(npz_labels))
        train_dataset = NPZDataset(npz_images[idx[:split]], npz_labels[idx[:split]])
        val_dataset = NPZDataset(npz_images[idx[split:]], npz_labels[idx[split:]])
    
    print(f"\n✓ Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
        num_workers=Config.NUM_WORKERS, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=True
    )
    
    # Train
    model, best_acc = train_model(train_loader, val_loader)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE")
    print(f"   Best Accuracy: {best_acc:.2f}%")
    print(f"   Model saved: {Config.OUTPUT_DIR / 'best_model.pth'}")
    print("="*60)


if __name__ == "__main__":
    main()
