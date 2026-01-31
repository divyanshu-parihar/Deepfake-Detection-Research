"""
Deepfake Detection - Complete Intra & Cross-Dataset Evaluation
===============================================================
Tests the model with all combinations of 4 datasets:
  - Synthetic (scikit-image based)
  - 140K Real/Fake Faces
  - Celeb-DF
  - FaceForensics++

Generates a full cross-dataset generalization matrix.

Uses kagglehub for dataset downloads (auto-installs if needed).

Usage in Colab:
    !git clone https://github.com/divyanshu-parihar/Deepfake-Detection-Research.git /content/repo
    %cd /content/repo
    %run colab_complete_eval.py
"""

import os
import sys
import glob
import random
import zipfile
import copy
from pathlib import Path
from typing import Optional, List, Dict

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
# Configuration
# =============================================================================

class Config:
    DATA_ROOT = Path('/content/data')
    CHECKPOINT_DIR = Path('/content/checkpoints')
    
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0005
    EPOCHS = 5
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMAGE_SIZE = 224
    MAX_SAMPLES = 5000  # Per class for real datasets
    
    @classmethod
    def setup_dirs(cls):
        cls.DATA_ROOT.mkdir(parents=True, exist_ok=True)
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

Config.setup_dirs()
random.seed(Config.SEED)
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)


# =============================================================================
# Dataset Downloads (using kagglehub)
# =============================================================================

def download_140k_faces() -> bool:
    """Download 140K Real/Fake Faces from Kaggle using kagglehub."""
    dataset_path = Config.DATA_ROOT / '140k_faces'
    dataset_path.mkdir(exist_ok=True)
    
    # Check if already exists
    if list(dataset_path.rglob('*.jpg'))[:1] or list(dataset_path.rglob('*.png'))[:1]:
        print("✓ 140K Faces dataset already exists")
        return True
    
    print("\nDownloading 140K Real/Fake Faces via kagglehub...")
    try:
        import kagglehub
        path = kagglehub.dataset_download("xhlulu/140k-real-and-fake-faces")
        print(f"✓ Downloaded to: {path}")
        
        # Create symlink or copy to our data directory
        import shutil
        if Path(path).exists():
            # Copy contents to our data path
            for item in Path(path).iterdir():
                dest = dataset_path / item.name
                if not dest.exists():
                    if item.is_dir():
                        shutil.copytree(item, dest)
                    else:
                        shutil.copy2(item, dest)
            print("✓ 140K Faces ready")
            return True
    except Exception as e:
        print(f"✗ Download failed: {e}")
    return False


def download_celeb_df() -> bool:
    """Download Celeb-DF images from Kaggle using kagglehub."""
    dataset_path = Config.DATA_ROOT / 'celeb_df'
    dataset_path.mkdir(exist_ok=True)
    
    if list(dataset_path.rglob('*.jpg'))[:1] or list(dataset_path.rglob('*.png'))[:1]:
        print("✓ Celeb-DF dataset already exists")
        return True
    
    print("\nDownloading Celeb-DF via kagglehub...")
    try:
        import kagglehub
        path = kagglehub.dataset_download("vksbhandary/celeb-df-ds-images")
        print(f"✓ Downloaded to: {path}")
        
        # Copy contents to our data path
        import shutil
        if Path(path).exists():
            for item in Path(path).iterdir():
                dest = dataset_path / item.name
                if not dest.exists():
                    if item.is_dir():
                        shutil.copytree(item, dest)
                    else:
                        shutil.copy2(item, dest)
            print("✓ Celeb-DF ready")
            return True
    except Exception as e:
        print(f"✗ Download failed: {e}")
    return False


def download_faceforensics() -> bool:
    """Download FaceForensics++ cropped images from Kaggle using kagglehub."""
    dataset_path = Config.DATA_ROOT / 'faceforensics'
    dataset_path.mkdir(exist_ok=True)
    
    if list(dataset_path.rglob('*.jpg'))[:1] or list(dataset_path.rglob('*.png'))[:1]:
        print("✓ FaceForensics++ dataset already exists")
        return True
    
    print("\nDownloading FaceForensics++ via kagglehub...")
    try:
        import kagglehub
        path = kagglehub.dataset_download("greatgamedota/faceforensics")
        print(f"✓ Downloaded to: {path}")
        
        # Copy contents to our data path
        import shutil
        if Path(path).exists():
            for item in Path(path).iterdir():
                dest = dataset_path / item.name
                if not dest.exists():
                    if item.is_dir():
                        shutil.copytree(item, dest)
                    else:
                        shutil.copy2(item, dest)
            print("✓ FaceForensics++ ready")
            return True
    except Exception as e:
        print(f"✗ Download failed: {e}")
    return False


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
# Dataset Classes
# =============================================================================

class SyntheticDataset(Dataset):
    """Synthetic deepfake dataset from scikit-image."""
    
    def __init__(self, mode='train', split_ratio=0.8):
        from skimage import data as skimage_data
        
        self.dct_processor = DCTProcessor()
        self.samples = []
        
        func_names = ['astronaut', 'camera', 'coffee', 'chelsea', 'cat', 'rocket', 'eagle']
        sources = []
        for name in func_names:
            if hasattr(skimage_data, name):
                try:
                    img = getattr(skimage_data, name)()
                    if len(img.shape) == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    sources.append(img)
                except:
                    pass
        
        all_patches = []
        for img in sources:
            h, w = img.shape
            for i in range(0, h - 112, 64):
                for j in range(0, w - 112, 64):
                    all_patches.append(img[i:i+112, j:j+112])
        
        random.shuffle(all_patches)
        split_idx = int(split_ratio * len(all_patches))
        self.samples = all_patches[:split_idx] if mode == 'train' else all_patches[split_idx:]
        print(f"  [SYNTHETIC-{mode.upper()}] {len(self.samples)*2} samples")
    
    def __len__(self):
        return len(self.samples) * 2
    
    def __getitem__(self, idx):
        base_idx = idx // 2
        is_fake = idx % 2
        img = self.samples[base_idx]
        h, w = img.shape
        
        if is_fake:
            algo = cv2.INTER_CUBIC if random.random() > 0.5 else cv2.INTER_NEAREST
            small = cv2.resize(img, (h//2, w//2), interpolation=cv2.INTER_AREA)
            img = cv2.resize(small, (h, w), interpolation=algo)
            label = 1
        else:
            label = 0
        
        spatial = cv2.resize(img, (224, 224)).astype(np.float32) / 127.5 - 1.0
        spatial = np.stack([spatial]*3, axis=0)
        
        dct = self.dct_processor.process_image(img)
        dct = cv2.resize(dct, (224, 224), interpolation=cv2.INTER_NEAREST)[np.newaxis]
        
        return {
            'spatial': torch.tensor(spatial).float(),
            'freq': torch.tensor(dct).float(),
            'label': torch.tensor(label).long()
        }


class RealFakeDataset(Dataset):
    """Generic real/fake image dataset loader."""
    
    def __init__(self, name: str, mode: str = 'train', split_ratio: float = 0.8):
        self.dct_processor = DCTProcessor()
        self.samples = []
        
        real_images = []
        fake_images = []
        
        if name == '140k_faces':
            base = Config.DATA_ROOT / '140k_faces'
            # Find real/fake directories
            for subdir in base.rglob('*'):
                if subdir.is_dir():
                    name_lower = subdir.name.lower()
                    if name_lower == 'real':
                        real_images.extend(list(subdir.glob('*.jpg')) + list(subdir.glob('*.png')))
                    elif name_lower == 'fake':
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
            # FaceForensics++ structure: cropped_images/XXX_YYY/
            # XXX = source video, YYY = target video
            # If XXX == YYY or only one number, it's original (real)
            # If XXX != YYY (two different numbers), it's manipulated (fake)
            for subdir in base.rglob('*'):
                if subdir.is_dir():
                    dir_name = subdir.name
                    # Check if it's a numbered directory like "000_003" or "cropped_images"
                    if '_' in dir_name and dir_name.replace('_', '').isdigit():
                        parts = dir_name.split('_')
                        if len(parts) == 2:
                            # XXX_YYY format - if different, it's manipulated
                            if parts[0] != parts[1]:
                                # Fake (manipulated)
                                fake_images.extend(list(subdir.glob('*.jpg')) + list(subdir.glob('*.png')))
                            else:
                                # Real (original)
                                real_images.extend(list(subdir.glob('*.jpg')) + list(subdir.glob('*.png')))
            
            # If no real images found with same numbers, use first half as real proxy
            if len(real_images) == 0 and len(fake_images) > 0:
                print("  Note: FF++ has mostly manipulated data. Using half as pseudo-real.")
                all_imgs = fake_images.copy()
                random.shuffle(all_imgs)
                mid = len(all_imgs) // 2
                real_images = all_imgs[:mid]
                fake_images = all_imgs[mid:]
        
        # Balance and limit
        random.shuffle(real_images)
        random.shuffle(fake_images)
        
        n_samples = min(len(real_images), len(fake_images), Config.MAX_SAMPLES)
        real_images = real_images[:n_samples]
        fake_images = fake_images[:n_samples]
        
        # Split
        split_idx = int(split_ratio * n_samples)
        if mode == 'train':
            self.samples = [(str(p), 0) for p in real_images[:split_idx]]
            self.samples += [(str(p), 1) for p in fake_images[:split_idx]]
        else:
            self.samples = [(str(p), 0) for p in real_images[split_idx:]]
            self.samples += [(str(p), 1) for p in fake_images[split_idx:]]
        
        random.shuffle(self.samples)
        print(f"  [{name.upper()}-{mode.upper()}] {len(self.samples)} samples")
    
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
            
            img = cv2.resize(img, (224, 224))
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            spatial = img.astype(np.float32) / 127.5 - 1.0
            spatial = spatial.transpose(2, 0, 1)
            
            dct = self.dct_processor.process_image(gray)
            dct = cv2.resize(dct, (224, 224), interpolation=cv2.INTER_NEAREST)[np.newaxis]
            
        except:
            spatial = np.zeros((3, 224, 224), dtype=np.float32)
            dct = np.zeros((1, 224, 224), dtype=np.float32)
        
        return {
            'spatial': torch.tensor(spatial).float(),
            'freq': torch.tensor(dct).float(),
            'label': torch.tensor(label).long()
        }


# =============================================================================
# Model
# =============================================================================

class DualStreamNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Frequency stream
        self.freq_conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.freq_bn1 = nn.BatchNorm2d(32)
        self.freq_conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.freq_bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.freq_fc = nn.LazyLinear(512)
        
        # RGB stream
        self.rgb_conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.rgb_bn1 = nn.BatchNorm2d(16)
        self.rgb_conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.rgb_bn2 = nn.BatchNorm2d(32)
        self.rgb_conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.rgb_bn3 = nn.BatchNorm2d(64)
        self.rgb_fc = nn.LazyLinear(512)
        
        # Fusion
        self.fusion = nn.Linear(1024, 128)
        self.classifier = nn.Linear(128, 2)
    
    def forward(self, x_rgb, x_freq):
        # Freq stream
        f = self.pool(F.relu(self.freq_bn1(self.freq_conv1(x_freq))))
        f = self.pool(F.relu(self.freq_bn2(self.freq_conv2(f))))
        f = f.view(f.size(0), -1)
        f = self.freq_fc(f)
        
        # RGB stream
        r = self.pool(F.relu(self.rgb_bn1(self.rgb_conv1(x_rgb))))
        r = self.pool(F.relu(self.rgb_bn2(self.rgb_conv2(r))))
        r = self.pool(F.relu(self.rgb_bn3(self.rgb_conv3(r))))
        r = r.view(r.size(0), -1)
        r = self.rgb_fc(r)
        
        # Fusion
        combined = torch.cat([r, f], dim=1)
        emb = F.relu(self.fusion(combined))
        out = self.classifier(emb)
        return out, emb


# =============================================================================
# Trainer
# =============================================================================

def create_model():
    """Create and initialize model."""
    model = DualStreamNetwork().to(Config.DEVICE)
    # Initialize lazy layers
    dummy_rgb = torch.randn(1, 3, 224, 224).to(Config.DEVICE)
    dummy_freq = torch.randn(1, 1, 224, 224).to(Config.DEVICE)
    model(dummy_rgb, dummy_freq)
    return model


def train_model(train_loader: DataLoader, epochs: int = 5) -> nn.Module:
    """Train a new model."""
    model = create_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        
        for batch in train_loader:
            spatial = batch['spatial'].to(Config.DEVICE)
            freq = batch['freq'].to(Config.DEVICE)
            labels = batch['label'].to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs, _ = model(spatial, freq)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        print(f"    Epoch {epoch+1}/{epochs} - Acc: {100*correct/total:.2f}%")
    
    return model


def evaluate_model(model: nn.Module, test_loader: DataLoader) -> Dict:
    """Evaluate model on test set."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            spatial = batch['spatial'].to(Config.DEVICE)
            freq = batch['freq'].to(Config.DEVICE)
            labels = batch['label'].to(Config.DEVICE)
            
            outputs, _ = model(spatial, freq)
            probs = F.softmax(outputs, dim=1)[:, 1]
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    acc = accuracy_score(all_labels, all_preds) * 100
    try:
        auc = roc_auc_score(all_labels, all_probs) * 100
    except:
        auc = 50.0
    f1 = f1_score(all_labels, all_preds, zero_division=0) * 100
    
    return {'accuracy': acc, 'auc': auc, 'f1': f1, 'samples': len(all_labels)}


# =============================================================================
# Main Evaluation Pipeline
# =============================================================================

def get_dataset(name: str, mode: str):
    """Get dataset by name."""
    if name == 'synthetic':
        return SyntheticDataset(mode)
    else:
        return RealFakeDataset(name, mode)


def run_full_evaluation():
    """Run complete intra and cross-dataset evaluation."""
    
    print("\n" + "="*70)
    print("DEEPFAKE DETECTION - COMPLETE EVALUATION MATRIX")
    print("="*70)
    print(f"Device: {Config.DEVICE}")
    print("="*70)
    
    # Download datasets
    print("\n📥 DOWNLOADING DATASETS...")
    datasets_available = {
        'synthetic': True,  # Always available
        '140k_faces': download_140k_faces(),
        'celeb_df': download_celeb_df(),
        'faceforensics': download_faceforensics(),
    }
    
    active_datasets = [k for k, v in datasets_available.items() if v]
    print(f"\n✓ Available datasets: {active_datasets}")
    
    # Results matrix
    results_matrix = {}
    
    # For each training dataset
    for train_name in active_datasets:
        print(f"\n{'='*70}")
        print(f"🎯 TRAINING ON: {train_name.upper()}")
        print("="*70)
        
        # Create train loader
        print("\n  Loading training data...")
        train_set = get_dataset(train_name, 'train')
        
        if len(train_set) == 0:
            print(f"  ⚠ No training samples for {train_name}, skipping...")
            continue
        
        train_loader = DataLoader(train_set, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2)
        
        # Train model
        print(f"\n  Training model ({Config.EPOCHS} epochs)...")
        model = train_model(train_loader, epochs=Config.EPOCHS)
        
        # Save model
        model_path = Config.CHECKPOINT_DIR / f'model_trained_on_{train_name}.pth'
        torch.save(model.state_dict(), model_path)
        print(f"  ✓ Model saved: {model_path.name}")
        
        results_matrix[train_name] = {}
        
        # Test on each dataset
        for test_name in active_datasets:
            print(f"\n  📊 Testing on: {test_name.upper()}")
            
            test_set = get_dataset(test_name, 'test')
            if len(test_set) == 0:
                print(f"    ⚠ No test samples for {test_name}")
                results_matrix[train_name][test_name] = {'accuracy': 0, 'auc': 0, 'f1': 0}
                continue
            
            test_loader = DataLoader(test_set, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)
            
            result = evaluate_model(model, test_loader)
            results_matrix[train_name][test_name] = result
            
            is_intra = train_name == test_name
            marker = "🟢 INTRA" if is_intra else "🔵 CROSS"
            print(f"    {marker} | Acc: {result['accuracy']:.2f}% | AUC: {result['auc']:.2f}% | F1: {result['f1']:.2f}%")
    
    # Print final results
    print_results_matrix(results_matrix, active_datasets)
    
    return results_matrix


def print_results_matrix(results: Dict, datasets: List[str]):
    """Print formatted results matrix."""
    
    print("\n" + "="*70)
    print("📊 COMPLETE RESULTS MATRIX")
    print("="*70)
    
    # Accuracy Matrix
    print("\n" + "-"*50)
    print("ACCURACY (%) - Rows=Train, Cols=Test")
    print("-"*50)
    
    # Header
    header = f"{'Train\\Test':<15}"
    for test_name in datasets:
        header += f"{test_name[:12]:>12}"
    print(header)
    print("-"*50)
    
    # Rows
    for train_name in datasets:
        if train_name not in results:
            continue
        row = f"{train_name:<15}"
        for test_name in datasets:
            if test_name in results.get(train_name, {}):
                acc = results[train_name][test_name]['accuracy']
                marker = "*" if train_name == test_name else " "
                row += f"{acc:>11.2f}{marker}"
            else:
                row += f"{'N/A':>12}"
        print(row)
    
    print("\n* = Intra-dataset (train and test on same dataset)")
    
    # AUC Matrix
    print("\n" + "-"*50)
    print("AUC-ROC (%) - Rows=Train, Cols=Test")
    print("-"*50)
    
    header = f"{'Train\\Test':<15}"
    for test_name in datasets:
        header += f"{test_name[:12]:>12}"
    print(header)
    print("-"*50)
    
    for train_name in datasets:
        if train_name not in results:
            continue
        row = f"{train_name:<15}"
        for test_name in datasets:
            if test_name in results.get(train_name, {}):
                auc = results[train_name][test_name]['auc']
                marker = "*" if train_name == test_name else " "
                row += f"{auc:>11.2f}{marker}"
            else:
                row += f"{'N/A':>12}"
        print(row)
    
    # Generalization Analysis
    print("\n" + "="*70)
    print("📈 GENERALIZATION ANALYSIS")
    print("="*70)
    
    for train_name in datasets:
        if train_name not in results:
            continue
        
        if train_name not in results[train_name]:
            continue
            
        intra_acc = results[train_name][train_name]['accuracy']
        
        print(f"\n{train_name.upper()} (Intra: {intra_acc:.2f}%)")
        for test_name in datasets:
            if test_name == train_name:
                continue
            if test_name not in results.get(train_name, {}):
                continue
            
            cross_acc = results[train_name][test_name]['accuracy']
            drop = intra_acc - cross_acc
            
            if drop > 20:
                status = "❌ Poor"
            elif drop > 10:
                status = "⚠️ Moderate"
            elif drop > 5:
                status = "🟡 Good"
            else:
                status = "✅ Excellent"
            
            print(f"  → {test_name}: {cross_acc:.2f}% (drop: {drop:+.2f}%) {status}")
    
    print("\n" + "="*70)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("📦 Installing required packages...")
    print("="*60)
    
    # Install kagglehub if not present
    try:
        import kagglehub
        print("✓ kagglehub already installed")
    except ImportError:
        import subprocess
        subprocess.check_call(['pip', 'install', 'kagglehub', '-q'])
        print("✓ kagglehub installed")
    
    try:
        from skimage import data as _
        print("✓ scikit-image already installed")
    except ImportError:
        import subprocess
        subprocess.check_call(['pip', 'install', 'scikit-image', '-q'])
        print("✓ scikit-image installed")
    
    results = run_full_evaluation()
    
    print("\n✅ Evaluation complete!")
    print(f"📁 Models saved to: {Config.CHECKPOINT_DIR}")

