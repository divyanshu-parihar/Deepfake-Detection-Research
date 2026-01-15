import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import random
from skimage import data
from src.core.dct import DCTProcessor

class SyntheticDeepfakeDataset(Dataset):
    """
    Generates a synthetic forensic dataset on-the-fly by slicing standard images
    and applying mathematical upsampling artifacts (Bicubic/Nearest) to simulate Deepfakes.
    """
    def __init__(self, mode='train', split_ratio=0.8, seed=999):
        self.samples = []
        self.dct_processor = DCTProcessor(block_size=8)
        
        # 1. Load Source Images (High-Res from scikit-image)
        # We try to load as many as possible to ensure diversity
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
        
        # 2. Generate Patches
        # Stride 64 provides a good balance of data volume vs redundancy
        patch_size = 112 
        stride = 64
        
        all_patches = []
        for img in sources:
            h, w = img.shape
            for i in range(0, h - patch_size, stride):
                for j in range(0, w - patch_size, stride):
                    patch = img[i:i+patch_size, j:j+patch_size]
                    all_patches.append(patch)
        
        # 3. Train/Test Split
        random.seed(seed)
        random.shuffle(all_patches)
        
        split_idx = int(split_ratio * len(all_patches))
        if mode == 'train':
            self.samples = all_patches[:split_idx]
        else:
            self.samples = all_patches[split_idx:]
            
        print(f"[{mode.upper()}] Dataset initialized with {len(self.samples) * 2} samples (Balanced).")

    def __len__(self):
        return len(self.samples) * 2 

    def __getitem__(self, idx):
        # Even idx = Real, Odd idx = Fake
        base_idx = idx // 2
        is_fake = idx % 2
        
        img = self.samples[base_idx]
        h, w = img.shape # (112, 112)
        
        if is_fake:
            # Simulate Generator Artifacts
            # Randomly choose between Bicubic (Blurry) and Nearest (Blocky) upsampling
            algo = cv2.INTER_CUBIC if random.random() > 0.5 else cv2.INTER_NEAREST
            
            # Downsample -> Upsample pipeline
            small = cv2.resize(img, (h//2, w//2), interpolation=cv2.INTER_AREA)
            img_processed = cv2.resize(small, (h, w), interpolation=algo)
            label = 1
        else:
            img_processed = img
            label = 0
            
        # --- Preprocessing for Network ---
        
        # 1. Spatial Stream (RGB)
        # Resize to standard CNN input (224x224)
        spatial = cv2.resize(img_processed, (224, 224))
        # Normalize [-1, 1]
        spatial = spatial.astype(np.float32) / 127.5 - 1.0 
        # Stack to 3 channels
        spatial = np.stack([spatial]*3, axis=-1) 
        # HWC -> CHW
        spatial = spatial.transpose(2, 0, 1) 
        
        # 2. Frequency Stream (DCT)
        # Process the raw patch to keep artifacts sharp
        dct_map = self.dct_processor.process_image(img_processed)
        # Resize map to CNN input
        dct_map = cv2.resize(dct_map, (224, 224), interpolation=cv2.INTER_NEAREST)
        dct_input = dct_map[np.newaxis, :, :] # Add channel dim -> (1, 224, 224)
        
        return {
            'spatial': torch.tensor(spatial).float(),
            'freq': torch.tensor(dct_input).float(),
            'label': torch.tensor(label).long()
        }