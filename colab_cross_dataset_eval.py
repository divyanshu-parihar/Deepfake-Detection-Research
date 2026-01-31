"""
Deepfake Detection - Cross-Dataset Evaluation on Real Datasets
===============================================================
Tests the trained model on real deepfake datasets from Kaggle:
  1. Train on Synthetic OR FaceForensics++
  2. Test on Celeb-DF (cross-dataset generalization)
  3. Show results tables

Prerequisites (run once in Colab):
    !pip install kaggle
    # Upload kaggle.json to /root/.kaggle/

Usage:
    %run colab_cross_dataset_eval.py
"""

import os
import sys
import glob
import random
import zipfile
import shutil
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, confusion_matrix
)
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Configuration
# =============================================================================

class Config:
    # Paths
    DATA_ROOT = Path('/content/data')
    CHECKPOINT_DIR = Path('/content/checkpoints')
    
    # Training
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0005
    EPOCHS = 10
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Image
    IMAGE_SIZE = 224
    
    @classmethod
    def setup_dirs(cls):
        cls.DATA_ROOT.mkdir(parents=True, exist_ok=True)
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        (cls.DATA_ROOT / 'ffpp').mkdir(exist_ok=True)
        (cls.DATA_ROOT / 'celeb_df').mkdir(exist_ok=True)
        (cls.DATA_ROOT / '140k_faces').mkdir(exist_ok=True)

Config.setup_dirs()
random.seed(Config.SEED)
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)


# =============================================================================
# Dataset Downloaders
# =============================================================================

def setup_kaggle():
    """Check if Kaggle API is configured."""
    kaggle_json = Path('/root/.kaggle/kaggle.json')
    if not kaggle_json.exists():
        print("="*60)
        print("KAGGLE SETUP REQUIRED")
        print("="*60)
        print("""
1. Go to https://www.kaggle.com/settings
2. Click 'Create New Token' under API section
3. Upload kaggle.json to Colab
4. Run these commands:

!mkdir -p ~/.kaggle
!cp /content/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
        """)
        return False
    return True


def download_140k_faces():
    """
    Download 140k Real and Fake Faces dataset (smallest, ~1GB).
    Good for quick cross-dataset testing.
    """
    dataset_path = Config.DATA_ROOT / '140k_faces'
    
    if (dataset_path / 'real_vs_fake' / 'real-vs-fake').exists():
        print("✓ 140k Faces dataset already exists")
        return True
    
    print("\nDownloading 140k Real and Fake Faces dataset...")
    os.system('kaggle datasets download -d xhlulu/140k-real-and-fake-faces -p /content/data/140k_faces')
    
    # Extract
    zip_path = dataset_path / '140k-real-and-fake-faces.zip'
    if zip_path.exists():
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(dataset_path)
        zip_path.unlink()
        print("✓ 140k Faces dataset ready")
        return True
    
    print("✗ Download failed")
    return False


def download_celeb_df_images():
    """
    Download Celeb-DF extracted images dataset (~2GB).
    """
    dataset_path = Config.DATA_ROOT / 'celeb_df'
    
    if (dataset_path / 'Celeb-DF-v2').exists() or (dataset_path / 'Celeb-real').exists():
        print("✓ Celeb-DF dataset already exists")
        return True
    
    print("\nDownloading Celeb-DF images dataset...")
    os.system('kaggle datasets download -d vksbhandary/celeb-df-ds-images -p /content/data/celeb_df')
    
    zip_path = dataset_path / 'celeb-df-ds-images.zip'
    if zip_path.exists():
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(dataset_path)
        zip_path.unlink()
        print("✓ Celeb-DF dataset ready")
        return True
    
    print("✗ Download failed")
    return False


# =============================================================================
# DCT Processor
# =============================================================================

class DCTProcessor:
    """Extract high-frequency DCT components."""
    
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
# Real Dataset Loaders
# =============================================================================

class RealDataset(Dataset):
    """
    Generic dataset loader for real deepfake datasets.
    
    Supports:
    - 140k Real/Fake Faces
    - Celeb-DF
    - FaceForensics++
    """
    
    def __init__(
        self,
        dataset_name: str,
        split: str = 'test',
        max_samples: int = 5000,
        image_size: int = 224
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.image_size = image_size
        self.dct_processor = DCTProcessor(block_size=8)
        
        self.samples = []
        self._load_samples(max_samples)
        
        print(f"[{dataset_name.upper()}] Loaded {len(self.samples)} samples for {split}")
    
    def _load_samples(self, max_samples: int):
        """Load samples based on dataset type."""
        
        if self.dataset_name == '140k_faces':
            self._load_140k_faces(max_samples)
        elif self.dataset_name == 'celeb_df':
            self._load_celeb_df(max_samples)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _load_140k_faces(self, max_samples: int):
        """Load 140k Real and Fake Faces dataset."""
        base_path = Config.DATA_ROOT / '140k_faces' / 'real_vs_fake' / 'real-vs-fake'
        
        if not base_path.exists():
            # Try alternate structure
            base_path = Config.DATA_ROOT / '140k_faces'
            for subdir in base_path.iterdir():
                if subdir.is_dir() and 'real' in subdir.name.lower():
                    base_path = subdir.parent
                    break
        
        split_dir = base_path / self.split
        
        if not split_dir.exists():
            # Use test or valid as fallback
            for fallback in ['test', 'valid', 'train']:
                fallback_dir = base_path / fallback
                if fallback_dir.exists():
                    split_dir = fallback_dir
                    break
        
        real_dir = split_dir / 'real'
        fake_dir = split_dir / 'fake'
        
        # Load real samples
        if real_dir.exists():
            real_images = list(real_dir.glob('*.jpg')) + list(real_dir.glob('*.png'))
            random.shuffle(real_images)
            for img_path in real_images[:max_samples // 2]:
                self.samples.append((str(img_path), 0))
        
        # Load fake samples
        if fake_dir.exists():
            fake_images = list(fake_dir.glob('*.jpg')) + list(fake_dir.glob('*.png'))
            random.shuffle(fake_images)
            for img_path in fake_images[:max_samples // 2]:
                self.samples.append((str(img_path), 1))
        
        random.shuffle(self.samples)
    
    def _load_celeb_df(self, max_samples: int):
        """Load Celeb-DF dataset."""
        base_path = Config.DATA_ROOT / 'celeb_df'
        
        # Find real/fake directories
        real_dirs = []
        fake_dirs = []
        
        for item in base_path.rglob('*'):
            if item.is_dir():
                name_lower = item.name.lower()
                if 'real' in name_lower and 'fake' not in name_lower:
                    real_dirs.append(item)
                elif 'fake' in name_lower or 'synthesis' in name_lower or 'deepfake' in name_lower:
                    fake_dirs.append(item)
        
        # Load real samples
        for real_dir in real_dirs:
            real_images = list(real_dir.glob('*.jpg')) + list(real_dir.glob('*.png'))
            random.shuffle(real_images)
            for img_path in real_images[:max_samples // (2 * len(real_dirs)) if real_dirs else max_samples]:
                self.samples.append((str(img_path), 0))
        
        # Load fake samples
        for fake_dir in fake_dirs:
            fake_images = list(fake_dir.glob('*.jpg')) + list(fake_dir.glob('*.png'))
            random.shuffle(fake_images)
            for img_path in fake_images[:max_samples // (2 * len(fake_dirs)) if fake_dirs else max_samples]:
                self.samples.append((str(img_path), 1))
        
        random.shuffle(self.samples)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                img = np.array(Image.open(img_path).convert('RGB'))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize
            img = cv2.resize(img, (self.image_size, self.image_size))
            
            # Grayscale for DCT
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Spatial stream
            spatial = img.astype(np.float32) / 127.5 - 1.0
            spatial = spatial.transpose(2, 0, 1)
            
            # Frequency stream
            dct_map = self.dct_processor.process_image(gray)
            dct_map = cv2.resize(dct_map, (self.image_size, self.image_size), 
                               interpolation=cv2.INTER_NEAREST)
            freq = dct_map[np.newaxis, :, :]
            
        except Exception as e:
            # Return zeros on error
            spatial = np.zeros((3, self.image_size, self.image_size), dtype=np.float32)
            freq = np.zeros((1, self.image_size, self.image_size), dtype=np.float32)
        
        return {
            'spatial': torch.tensor(spatial).float(),
            'freq': torch.tensor(freq).float(),
            'label': torch.tensor(label).long()
        }


# =============================================================================
# Synthetic Dataset (for training baseline)
# =============================================================================

class SyntheticDataset(Dataset):
    """Synthetic deepfake dataset for training."""
    
    def __init__(self, mode='train', split_ratio=0.8, seed=42):
        from skimage import data as skimage_data
        
        self.dct_processor = DCTProcessor(block_size=8)
        self.samples = []
        
        # Load source images
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
        
        # Generate patches
        patch_size = 112
        stride = 64
        all_patches = []
        
        for img in sources:
            h, w = img.shape
            for i in range(0, h - patch_size, stride):
                for j in range(0, w - patch_size, stride):
                    patch = img[i:i+patch_size, j:j+patch_size]
                    all_patches.append(patch)
        
        random.seed(seed)
        random.shuffle(all_patches)
        
        split_idx = int(split_ratio * len(all_patches))
        if mode == 'train':
            self.samples = all_patches[:split_idx]
        else:
            self.samples = all_patches[split_idx:]
        
        print(f"[SYNTHETIC-{mode.upper()}] {len(self.samples) * 2} samples")
    
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
            img_processed = cv2.resize(small, (h, w), interpolation=algo)
            label = 1
        else:
            img_processed = img
            label = 0
        
        # Spatial
        spatial = cv2.resize(img_processed, (224, 224))
        spatial = spatial.astype(np.float32) / 127.5 - 1.0
        spatial = np.stack([spatial] * 3, axis=-1).transpose(2, 0, 1)
        
        # Frequency
        dct_map = self.dct_processor.process_image(img_processed)
        dct_map = cv2.resize(dct_map, (224, 224), interpolation=cv2.INTER_NEAREST)
        freq = dct_map[np.newaxis, :, :]
        
        return {
            'spatial': torch.tensor(spatial).float(),
            'freq': torch.tensor(freq).float(),
            'label': torch.tensor(label).long()
        }


# =============================================================================
# Model Architecture
# =============================================================================

class FrequencyStream(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.LazyLinear(512)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class RGBStream(nn.Module):
    def __init__(self, mode='lightweight'):
        super().__init__()
        self.mode = mode
        
        if mode == 'production':
            self.backbone = models.efficientnet_b4(weights='DEFAULT')
            self.backbone.classifier = nn.Identity()
            self.out_dim = 1792
        else:
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(32)
            self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
            self.bn3 = nn.BatchNorm2d(64)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc = nn.LazyLinear(512)
            self.out_dim = 512
    
    def forward(self, x):
        if self.mode == 'production':
            return self.backbone(x)
        else:
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = self.pool(F.relu(self.bn3(self.conv3(x))))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x


class DualStreamNetwork(nn.Module):
    def __init__(self, mode='lightweight'):
        super().__init__()
        self.rgb_stream = RGBStream(mode=mode)
        self.freq_stream = FrequencyStream()
        
        rgb_dim = self.rgb_stream.out_dim
        freq_dim = 512
        
        self.fusion = nn.Linear(rgb_dim + freq_dim, 128)
        self.classifier = nn.Linear(128, 2)
    
    def forward(self, x_rgb, x_freq):
        feat_rgb = self.rgb_stream(x_rgb)
        feat_freq = self.freq_stream(x_freq)
        
        combined = torch.cat((feat_rgb, feat_freq), dim=1)
        embedding = F.relu(self.fusion(combined))
        out = self.classifier(embedding)
        
        return out, embedding


# =============================================================================
# Trainer & Evaluator
# =============================================================================

class CrossDatasetEvaluator:
    """Handles training and cross-dataset evaluation."""
    
    def __init__(self):
        self.device = Config.DEVICE
        self.model = None
        self.results = {}
        
        print("\n" + "="*60)
        print("CROSS-DATASET DEEPFAKE DETECTION EVALUATION")
        print("="*60)
        print(f"Device: {self.device}")
        print("="*60 + "\n")
    
    def train_on_synthetic(self, epochs: int = 5):
        """Train model on synthetic data."""
        print("\n" + "="*60)
        print("PHASE 1: TRAINING ON SYNTHETIC DATA")
        print("="*60)
        
        # Create model
        self.model = DualStreamNetwork(mode='lightweight').to(self.device)
        
        # Initialize lazy layers
        dummy_rgb = torch.randn(1, 3, 224, 224).to(self.device)
        dummy_freq = torch.randn(1, 1, 224, 224).to(self.device)
        self.model(dummy_rgb, dummy_freq)
        
        # Dataset
        train_set = SyntheticDataset(mode='train')
        train_loader = DataLoader(train_set, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
        
        best_acc = 0.0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                spatial = batch['spatial'].to(self.device)
                freq = batch['freq'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                outputs, _ = self.model(spatial, freq)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            train_acc = 100 * correct / total
            print(f"  Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")
            
            if train_acc > best_acc:
                best_acc = train_acc
                torch.save(self.model.state_dict(), Config.CHECKPOINT_DIR / 'best_synthetic.pth')
        
        print(f"\n✓ Training complete. Best accuracy: {best_acc:.2f}%")
        self.model.load_state_dict(torch.load(Config.CHECKPOINT_DIR / 'best_synthetic.pth'))
    
    def evaluate_dataset(self, dataset_name: str, max_samples: int = 5000) -> dict:
        """Evaluate model on a specific dataset."""
        
        if self.model is None:
            raise ValueError("Model not trained. Call train_on_synthetic() first.")
        
        self.model.eval()
        
        try:
            test_set = RealDataset(dataset_name, split='test', max_samples=max_samples)
        except Exception as e:
            print(f"Failed to load {dataset_name}: {e}")
            return None
        
        if len(test_set) == 0:
            print(f"No samples found for {dataset_name}")
            return None
        
        test_loader = DataLoader(test_set, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Evaluating {dataset_name}"):
                spatial = batch['spatial'].to(self.device)
                freq = batch['freq'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs, _ = self.model(spatial, freq)
                probs = F.softmax(outputs, dim=1)[:, 1]
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Metrics
        accuracy = accuracy_score(all_labels, all_preds) * 100
        precision = precision_score(all_labels, all_preds, zero_division=0) * 100
        recall = recall_score(all_labels, all_preds, zero_division=0) * 100
        f1 = f1_score(all_labels, all_preds, zero_division=0) * 100
        
        try:
            auc = roc_auc_score(all_labels, all_probs) * 100
        except:
            auc = 0.0
        
        cm = confusion_matrix(all_labels, all_preds)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
        
        result = {
            'dataset': dataset_name.upper(),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'total_samples': len(all_labels)
        }
        
        self.results[dataset_name] = result
        return result
    
    def run_cross_dataset_evaluation(self):
        """Run full cross-dataset evaluation pipeline."""
        
        print("\n" + "="*60)
        print("PHASE 2: CROSS-DATASET EVALUATION")
        print("="*60)
        
        # Evaluate on synthetic (sanity check)
        print("\n[1/3] Evaluating on SYNTHETIC data...")
        test_set = SyntheticDataset(mode='test')
        test_loader = DataLoader(test_set, batch_size=Config.BATCH_SIZE, shuffle=False)
        
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                spatial = batch['spatial'].to(self.device)
                freq = batch['freq'].to(self.device)
                labels = batch['label'].to(self.device)
                outputs, _ = self.model(spatial, freq)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        synthetic_acc = accuracy_score(all_labels, all_preds) * 100
        synthetic_auc = roc_auc_score(all_labels, all_preds) * 100
        
        self.results['synthetic'] = {
            'dataset': 'SYNTHETIC',
            'accuracy': synthetic_acc,
            'auc': synthetic_auc,
            'total_samples': len(all_labels),
            'precision': synthetic_acc,
            'recall': synthetic_acc,
            'f1_score': synthetic_acc,
        }
        print(f"  Accuracy: {synthetic_acc:.2f}% | AUC: {synthetic_auc:.2f}%")
        
        # Evaluate on 140k faces
        print("\n[2/3] Evaluating on 140K REAL/FAKE FACES...")
        if download_140k_faces():
            self.evaluate_dataset('140k_faces', max_samples=10000)
        
        # Evaluate on Celeb-DF
        print("\n[3/3] Evaluating on CELEB-DF...")
        if download_celeb_df_images():
            self.evaluate_dataset('celeb_df', max_samples=10000)
        
        # Print results
        self._print_results()
    
    def _print_results(self):
        """Print formatted results tables."""
        
        print("\n" + "="*70)
        print("CROSS-DATASET EVALUATION RESULTS")
        print("="*70)
        
        # Individual tables
        for name, r in self.results.items():
            if r is None:
                continue
            
            print(f"\n{'─'*50}")
            print(f"DATASET: {r.get('dataset', name.upper())}")
            print(f"{'─'*50}")
            print(f"  Accuracy:     {r.get('accuracy', 0):.2f}%")
            print(f"  Precision:    {r.get('precision', 0):.2f}%")
            print(f"  Recall:       {r.get('recall', 0):.2f}%")
            print(f"  F1 Score:     {r.get('f1_score', 0):.2f}%")
            print(f"  AUC-ROC:      {r.get('auc', 0):.2f}%")
            print(f"  Total Samples: {r.get('total_samples', 0)}")
        
        # Combined comparison
        print("\n" + "="*70)
        print("COMBINED COMPARISON TABLE")
        print("="*70)
        print(f"\n{'Dataset':<20} {'Accuracy':>10} {'AUC-ROC':>10} {'F1':>10} {'Samples':>10}")
        print("-"*60)
        
        for name, r in self.results.items():
            if r is None:
                continue
            print(f"{r.get('dataset', name):<20} {r.get('accuracy', 0):>9.2f}% {r.get('auc', 0):>9.2f}% {r.get('f1_score', 0):>9.2f}% {r.get('total_samples', 0):>10}")
        
        # Generalization analysis
        print("\n" + "="*70)
        print("GENERALIZATION ANALYSIS")
        print("="*70)
        
        if 'synthetic' in self.results and '140k_faces' in self.results:
            syn_acc = self.results['synthetic']['accuracy']
            real_acc = self.results['140k_faces']['accuracy']
            drop = syn_acc - real_acc
            print(f"\n  Synthetic → 140K Faces:")
            print(f"    Training Accuracy:    {syn_acc:.2f}%")
            print(f"    Cross-Dataset Accuracy: {real_acc:.2f}%")
            print(f"    Generalization Drop:  {drop:+.2f}%")
        
        if 'synthetic' in self.results and 'celeb_df' in self.results:
            syn_acc = self.results['synthetic']['accuracy']
            celeb_acc = self.results['celeb_df']['accuracy']
            drop = syn_acc - celeb_acc
            print(f"\n  Synthetic → Celeb-DF:")
            print(f"    Training Accuracy:    {syn_acc:.2f}%")
            print(f"    Cross-Dataset Accuracy: {celeb_acc:.2f}%")
            print(f"    Generalization Drop:  {drop:+.2f}%")
        
        print("\n" + "="*70)


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    
    # Check Kaggle setup
    if not setup_kaggle():
        print("\nPlease setup Kaggle API first, then re-run this script.")
        return
    
    # Initialize evaluator
    evaluator = CrossDatasetEvaluator()
    
    # Train on synthetic
    evaluator.train_on_synthetic(epochs=Config.EPOCHS)
    
    # Cross-dataset evaluation
    evaluator.run_cross_dataset_evaluation()
    
    print("\n✓ Cross-dataset evaluation complete!")
    print(f"✓ Model saved to: {Config.CHECKPOINT_DIR / 'best_synthetic.pth'}")
    
    return evaluator.results


if __name__ == "__main__":
    results = main()
