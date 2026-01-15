import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
from skimage import data
from sklearn.metrics import accuracy_score, roc_auc_score
from demo_implementation import DualStreamNetwork, DCTProcessor
import random
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- 1. The Robust Synthetic Dataset ---
class SyntheticDeepfakeDataset(Dataset):
    def __init__(self, mode='train', transform=None):
        self.samples = []
        self.dct_processor = DCTProcessor(block_size=8)
        
        # 1. Load MORE Source Data
        # We use every available high-res image in skimage
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
        
        # 2. Generate Patches (Stride 32 = Maximum Overlap = More Data)
        patch_size = 112 # Smaller patches = More data = Faster training
        stride = 64
        
        print(f"[{mode.upper()}] Mining patches from {len(sources)} source images...")
        
        all_patches = []
        for img in sources:
            h, w = img.shape
            for i in range(0, h - patch_size, stride):
                for j in range(0, w - patch_size, stride):
                    patch = img[i:i+patch_size, j:j+patch_size]
                    all_patches.append(patch)
        
        # 3. Deterministic Split
        # Shuffle with seed so Train/Test don't leak
        random.seed(999)
        random.shuffle(all_patches)
        
        split_point = int(0.8 * len(all_patches))
        if mode == 'train':
            self.samples = all_patches[:split_point]
        else:
            self.samples = all_patches[split_point:]
            
        print(f"[{mode.upper()}] Dataset Size: {len(self.samples) * 2} items (Balanced Real/Fake)")

    def __len__(self):
        return len(self.samples) * 2 

    def __getitem__(self, idx):
        # Even = Real, Odd = Fake
        base_idx = idx // 2
        is_fake = idx % 2
        
        img = self.samples[base_idx]
        
        if is_fake:
            # --- THE GENERATOR ---
            # Simulate "Upsampling Artifacts"
            # 1. Downsample (Lose information)
            # 2. Upsample (Network has to 'guess' pixels -> Spectral Noise)
            h, w = img.shape
            
            # Mix artifacts: 50% Bicubic (Blurry), 50% Nearest (Blocky)
            # This forces the model to learn GENERAL anomalies, not just one type.
            algo = cv2.INTER_CUBIC if random.random() > 0.5 else cv2.INTER_NEAREST
            
            small = cv2.resize(img, (h//2, w//2), interpolation=cv2.INTER_AREA)
            img_processed = cv2.resize(small, (h, w), interpolation=algo)
            label = 1
        else:
            img_processed = img
            label = 0
            
        # --- PREPROCESSING (CRITICAL FOR ACCURACY) ---
        
        # 1. Spatial Stream (Resize to 224 for CNN)
        # Normalize to [-1, 1] for faster convergence
        spatial = cv2.resize(img_processed, (224, 224))
        spatial = spatial.astype(np.float32) / 127.5 - 1.0 
        spatial = np.stack([spatial]*3, axis=-1) # (224, 224, 3)
        spatial = spatial.transpose(2, 0, 1) # (3, 224, 224)
        
        # 2. Frequency Stream (DCT)
        # Use the raw patch for DCT (don't resize, keep artifacts pure)
        # We resize the MAP, not the IMAGE, to fit the CNN
        dct_map = self.dct_processor.process_image(img_processed)
        
        # Log-Normalization (The "Secret Sauce" to fix convergence)
        # DCT values are huge. We log-squash them to [0, 1].
        dct_map = np.log(np.abs(dct_map) + 1e-6)
        
        # Min-Max Scale per image (Instance Normalization)
        d_min, d_max = dct_map.min(), dct_map.max()
        if d_max - d_min > 0:
            dct_map = (dct_map - d_min) / (d_max - d_min)
        else:
            dct_map = np.zeros_like(dct_map)
            
        # Resize map to CNN input size
        dct_map = cv2.resize(dct_map, (224, 224), interpolation=cv2.INTER_NEAREST)
        dct_input = dct_map[np.newaxis, :, :] # (1, 224, 224)
        
        return {
            'spatial': torch.tensor(spatial).float(),
            'freq': torch.tensor(dct_input).float(),
            'label': torch.tensor(label).long()
        }

# --- 2. The Training Engine ---
def run_experiment():
    print("\n=== STARTING HIGH-FIDELITY EXPERIMENT ===")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")
    
    # Create Save Directory
    if not os.path.exists('models'):
        os.makedirs('models')
        
    train_dataset = SyntheticDeepfakeDataset(mode='train')
    test_dataset = SyntheticDeepfakeDataset(mode='test')
    
    # Batch size 32 is stable
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = DualStreamNetwork().to(device)
    
    # Run Dummy pass to init LazyLayers
    model(torch.randn(1, 3, 224, 224), torch.randn(1, 1, 224, 224))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005) # Lower LR for stability
    
    best_acc = 0.0
    epochs = 4 # Increased epochs
    
    print(f"\nTraining on {len(train_dataset)} samples for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch in train_loader:
            spatial = batch['spatial'].to(device)
            freq = batch['freq'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(spatial, freq)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        print(f"[Epoch {epoch+1}] Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")
        
        # --- TEST AND SAVE ---
        model.eval()
        test_correct = 0
        test_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                spatial = batch['spatial'].to(device)
                freq = batch['freq'].to(device)
                labels = batch['label'].to(device)
                
                outputs, _ = model(spatial, freq)
                _, predicted = torch.max(outputs.data, 1)
                
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        test_acc = 100 * test_correct / test_total
        auc = roc_auc_score(all_labels, all_preds)
        
        print(f"          Test Acc:  {test_acc:.2f}% | AUC: {auc:.4f}")
        
        if test_acc > best_acc:
            best_acc = test_acc
            save_path = "models/best_deepfake_detector.pth"
            torch.save(model.state_dict(), save_path)
            print(f"          >>> NEW RECORD! Model saved to {save_path}")

    print("\n" + "="*40)
    print(f"FINAL RESULT: {best_acc:.2f}% Accuracy")
    print("="*40)

if __name__ == "__main__":
    run_experiment()