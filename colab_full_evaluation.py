"""
Deepfake Detection - Complete Colab Training & Evaluation Script
=================================================================
Trains the dual-stream model and evaluates on both upsampling artifact types:
- Bicubic (blurry artifacts)
- Nearest Neighbor (blocky artifacts)

Outputs separate accuracy tables for each type + combined results.

Usage in Colab:
    1. Upload this file OR clone repo
    2. Run: %run colab_full_evaluation.py
"""

import sys
import os
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from skimage import data
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

class Config:
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0005
    EPOCHS = 5
    SEED = 999
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PATCH_SIZE = 112
    STRIDE = 64
    IMAGE_SIZE = 224
    
Config.SEED = 999
random.seed(Config.SEED)
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)


# =============================================================================
# DCT Processor
# =============================================================================

class DCTProcessor:
    """Extracts DCT high-frequency residuals to expose upsampling artifacts."""
    
    def __init__(self, block_size: int = 8):
        self.block_size = block_size
    
    def process_image(self, img: np.ndarray) -> np.ndarray:
        """Extract high-frequency DCT components."""
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        img = img.astype(np.float32)
        h, w = img.shape
        
        # Pad to block size multiple
        pad_h = (self.block_size - h % self.block_size) % self.block_size
        pad_w = (self.block_size - w % self.block_size) % self.block_size
        img_padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
        
        h_pad, w_pad = img_padded.shape
        output = np.zeros_like(img_padded)
        
        for i in range(0, h_pad, self.block_size):
            for j in range(0, w_pad, self.block_size):
                block = img_padded[i:i+self.block_size, j:j+self.block_size]
                dct_block = cv2.dct(block)
                
                # High-pass: zero out low frequencies (top-left quadrant)
                mask = np.ones_like(dct_block)
                mask[:self.block_size//2, :self.block_size//2] = 0
                dct_block *= mask
                
                output[i:i+self.block_size, j:j+self.block_size] = np.abs(dct_block)
        
        # Normalize
        output = output[:h, :w]
        if output.max() > 0:
            output = (output - output.min()) / (output.max() - output.min())
        
        return output.astype(np.float32)


# =============================================================================
# Datasets - Separate by Artifact Type
# =============================================================================

class ArtifactSpecificDataset(Dataset):
    """
    Dataset that generates specific artifact types for controlled evaluation.
    
    Args:
        mode: 'train' or 'test'
        artifact_type: 'bicubic', 'nearest', or 'mixed'
    """
    
    def __init__(self, mode='test', artifact_type='mixed', split_ratio=0.8, seed=999):
        self.artifact_type = artifact_type
        self.samples = []
        self.dct_processor = DCTProcessor(block_size=8)
        
        # Load source images from scikit-image
        func_names = ['astronaut', 'camera', 'coffee', 'chelsea', 'cat', 'rocket', 'eagle']
        sources = []
        for name in func_names:
            if hasattr(data, name):
                try:
                    img = getattr(data, name)()
                    if len(img.shape) == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    sources.append(img)
                except:
                    pass
        
        # Generate patches
        all_patches = []
        for img in sources:
            h, w = img.shape
            for i in range(0, h - Config.PATCH_SIZE, Config.STRIDE):
                for j in range(0, w - Config.PATCH_SIZE, Config.STRIDE):
                    patch = img[i:i+Config.PATCH_SIZE, j:j+Config.PATCH_SIZE]
                    all_patches.append(patch)
        
        # Train/Test split
        random.seed(seed)
        random.shuffle(all_patches)
        
        split_idx = int(split_ratio * len(all_patches))
        if mode == 'train':
            self.samples = all_patches[:split_idx]
        else:
            self.samples = all_patches[split_idx:]
        
        print(f"[{mode.upper()}] {artifact_type.upper()} Dataset: {len(self.samples) * 2} samples")
    
    def __len__(self):
        return len(self.samples) * 2
    
    def __getitem__(self, idx):
        base_idx = idx // 2
        is_fake = idx % 2
        
        img = self.samples[base_idx]
        h, w = img.shape
        
        if is_fake:
            # Apply specific artifact type
            if self.artifact_type == 'bicubic':
                algo = cv2.INTER_CUBIC
            elif self.artifact_type == 'nearest':
                algo = cv2.INTER_NEAREST
            else:  # mixed
                algo = cv2.INTER_CUBIC if random.random() > 0.5 else cv2.INTER_NEAREST
            
            small = cv2.resize(img, (h//2, w//2), interpolation=cv2.INTER_AREA)
            img_processed = cv2.resize(small, (h, w), interpolation=algo)
            label = 1
        else:
            img_processed = img
            label = 0
        
        # Spatial stream (RGB-like)
        spatial = cv2.resize(img_processed, (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
        spatial = spatial.astype(np.float32) / 127.5 - 1.0
        spatial = np.stack([spatial] * 3, axis=-1)
        spatial = spatial.transpose(2, 0, 1)
        
        # Frequency stream (DCT)
        dct_map = self.dct_processor.process_image(img_processed)
        dct_map = cv2.resize(dct_map, (Config.IMAGE_SIZE, Config.IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
        dct_input = dct_map[np.newaxis, :, :]
        
        return {
            'spatial': torch.tensor(spatial).float(),
            'freq': torch.tensor(dct_input).float(),
            'label': torch.tensor(label).long()
        }


# =============================================================================
# Model Architecture
# =============================================================================

class FrequencyStream(nn.Module):
    """CNN for processing DCT frequency maps."""
    
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
    """Spatial/semantic feature extractor."""
    
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
    """Dual-stream deepfake detector combining RGB and frequency features."""
    
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
# Trainer
# =============================================================================

class DeepfakeTrainer:
    """Training and evaluation orchestrator."""
    
    def __init__(self):
        self.device = Config.DEVICE
        print(f"\n{'='*60}")
        print(f"Deepfake Detection - Training & Evaluation")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {Config.EPOCHS}")
        print(f"Batch Size: {Config.BATCH_SIZE}")
        print(f"Learning Rate: {Config.LEARNING_RATE}")
        print(f"{'='*60}\n")
        
        # Training uses mixed artifacts
        self.train_set = ArtifactSpecificDataset(mode='train', artifact_type='mixed')
        self.train_loader = DataLoader(
            self.train_set, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=True,
            num_workers=2
        )
        
        # Model
        self.model = DualStreamNetwork(mode='lightweight').to(self.device)
        
        # Initialize lazy layers
        dummy_rgb = torch.randn(1, 3, 224, 224).to(self.device)
        dummy_freq = torch.randn(1, 1, 224, 224).to(self.device)
        self.model(dummy_rgb, dummy_freq)
        
        # Training setup
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model Parameters: {total_params:,} (Trainable: {trainable_params:,})")
        
        # Ensure models directory exists
        os.makedirs('models', exist_ok=True)
    
    def train(self):
        """Train the model."""
        print(f"\n{'='*60}")
        print("PHASE 1: TRAINING")
        print(f"{'='*60}\n")
        
        best_acc = 0.0
        
        for epoch in range(Config.EPOCHS):
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for batch in self.train_loader:
                spatial = batch['spatial'].to(self.device)
                freq = batch['freq'].to(self.device)
                labels = batch['label'].to(self.device)
                
                self.optimizer.zero_grad()
                outputs, _ = self.model(spatial, freq)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            train_acc = 100 * correct / total
            avg_loss = train_loss / len(self.train_loader)
            
            # Quick validation on mixed test set
            test_acc, test_auc = self._quick_eval()
            
            print(f"Epoch [{epoch+1}/{Config.EPOCHS}] "
                  f"Loss: {avg_loss:.4f} | "
                  f"Train Acc: {train_acc:.2f}% | "
                  f"Test Acc: {test_acc:.2f}% | "
                  f"AUC: {test_auc:.4f}")
            
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(self.model.state_dict(), "models/best_deepfake_detector.pth")
                print(f"          >>> NEW BEST! Model saved.")
        
        print(f"\nTraining Complete. Best Accuracy: {best_acc:.2f}%")
        
        # Load best model for evaluation
        self.model.load_state_dict(torch.load("models/best_deepfake_detector.pth"))
    
    def _quick_eval(self):
        """Quick evaluation on mixed test set."""
        self.model.eval()
        test_set = ArtifactSpecificDataset(mode='test', artifact_type='mixed')
        test_loader = DataLoader(test_set, batch_size=Config.BATCH_SIZE, shuffle=False)
        
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                spatial = batch['spatial'].to(self.device)
                freq = batch['freq'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs, _ = self.model(spatial, freq)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        acc = 100 * correct / total
        auc = roc_auc_score(all_labels, all_preds)
        return acc, auc
    
    def evaluate_artifact_type(self, artifact_type: str) -> dict:
        """Evaluate model on specific artifact type."""
        self.model.eval()
        
        test_set = ArtifactSpecificDataset(mode='test', artifact_type=artifact_type)
        test_loader = DataLoader(test_set, batch_size=Config.BATCH_SIZE, shuffle=False)
        
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
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds) * 100
        precision = precision_score(all_labels, all_preds, zero_division=0) * 100
        recall = recall_score(all_labels, all_preds, zero_division=0) * 100
        f1 = f1_score(all_labels, all_preds, zero_division=0) * 100
        auc = roc_auc_score(all_labels, all_probs) * 100
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        tn, fp, fn, tp = cm.ravel()
        
        return {
            'artifact_type': artifact_type.upper(),
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
    
    def full_evaluation(self):
        """Run full evaluation on all artifact types."""
        print(f"\n{'='*60}")
        print("PHASE 2: EVALUATION BY ARTIFACT TYPE")
        print(f"{'='*60}\n")
        
        results = {}
        
        # Evaluate each artifact type
        for artifact_type in ['bicubic', 'nearest', 'mixed']:
            print(f"\nEvaluating on {artifact_type.upper()} artifacts...")
            results[artifact_type] = self.evaluate_artifact_type(artifact_type)
        
        # Print results tables
        self._print_results_table(results)
        
        return results
    
    def _print_results_table(self, results: dict):
        """Print formatted results tables."""
        
        # Individual tables for each type
        for artifact_type in ['bicubic', 'nearest']:
            r = results[artifact_type]
            print(f"\n{'='*60}")
            print(f"RESULTS: {r['artifact_type']} ARTIFACTS")
            print(f"{'='*60}")
            print(f"┌{'─'*30}┬{'─'*15}┐")
            print(f"│ {'Metric':<28} │ {'Value':>13} │")
            print(f"├{'─'*30}┼{'─'*15}┤")
            print(f"│ {'Accuracy':<28} │ {r['accuracy']:>12.2f}% │")
            print(f"│ {'Precision':<28} │ {r['precision']:>12.2f}% │")
            print(f"│ {'Recall':<28} │ {r['recall']:>12.2f}% │")
            print(f"│ {'F1 Score':<28} │ {r['f1_score']:>12.2f}% │")
            print(f"│ {'AUC-ROC':<28} │ {r['auc']:>12.2f}% │")
            print(f"├{'─'*30}┼{'─'*15}┤")
            print(f"│ {'True Positives (Fake→Fake)':<28} │ {r['true_positives']:>13} │")
            print(f"│ {'True Negatives (Real→Real)':<28} │ {r['true_negatives']:>13} │")
            print(f"│ {'False Positives (Real→Fake)':<28} │ {r['false_positives']:>13} │")
            print(f"│ {'False Negatives (Fake→Real)':<28} │ {r['false_negatives']:>13} │")
            print(f"│ {'Total Samples':<28} │ {r['total_samples']:>13} │")
            print(f"└{'─'*30}┴{'─'*15}┘")
        
        # Combined comparison table
        print(f"\n{'='*60}")
        print("COMBINED COMPARISON TABLE")
        print(f"{'='*60}")
        print(f"┌{'─'*15}┬{'─'*12}┬{'─'*12}┬{'─'*12}┬{'─'*12}┬{'─'*12}┐")
        print(f"│ {'Artifact':<13} │ {'Accuracy':>10} │ {'Precision':>10} │ {'Recall':>10} │ {'F1 Score':>10} │ {'AUC-ROC':>10} │")
        print(f"├{'─'*15}┼{'─'*12}┼{'─'*12}┼{'─'*12}┼{'─'*12}┼{'─'*12}┤")
        
        for artifact_type in ['bicubic', 'nearest', 'mixed']:
            r = results[artifact_type]
            print(f"│ {artifact_type.upper():<13} │ {r['accuracy']:>9.2f}% │ {r['precision']:>9.2f}% │ {r['recall']:>9.2f}% │ {r['f1_score']:>9.2f}% │ {r['auc']:>9.2f}% │")
        
        print(f"└{'─'*15}┴{'─'*12}┴{'─'*12}┴{'─'*12}┴{'─'*12}┴{'─'*12}┘")
        
        # Summary statistics
        bicubic = results['bicubic']
        nearest = results['nearest']
        
        avg_acc = (bicubic['accuracy'] + nearest['accuracy']) / 2
        avg_auc = (bicubic['auc'] + nearest['auc']) / 2
        
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"  • Bicubic Accuracy:    {bicubic['accuracy']:.2f}%")
        print(f"  • Nearest Accuracy:    {nearest['accuracy']:.2f}%")
        print(f"  • Average Accuracy:    {avg_acc:.2f}%")
        print(f"  • Average AUC-ROC:     {avg_auc:.2f}%")
        print(f"  • Accuracy Difference: {abs(bicubic['accuracy'] - nearest['accuracy']):.2f}%")
        print(f"{'='*60}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run complete training and evaluation pipeline."""
    
    print("\n" + "="*60)
    print("DEEPFAKE DETECTION - COMPLETE EVALUATION PIPELINE")
    print("="*60)
    print("This script will:")
    print("  1. Train a dual-stream detector on mixed synthetic data")
    print("  2. Evaluate separately on Bicubic artifacts")
    print("  3. Evaluate separately on Nearest Neighbor artifacts")
    print("  4. Show comparison tables")
    print("="*60 + "\n")
    
    # Initialize and train
    trainer = DeepfakeTrainer()
    trainer.train()
    
    # Full evaluation
    results = trainer.full_evaluation()
    
    print("\n✓ Evaluation complete!")
    print(f"✓ Model saved to: models/best_deepfake_detector.pth")
    
    return results


if __name__ == "__main__":
    results = main()
