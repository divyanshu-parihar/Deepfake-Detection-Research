import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision import models
from PIL import Image
import scipy.fftpack
from skimage import data

# --- PART 1: The Core Math (DCT Processing) ---
class DCTProcessor:
    """
    Implements the artifact-invariant feature extraction described in Section III.
    """
    def __init__(self, block_size=8):
        self.block_size = block_size

    def process_image(self, image_gray):
        """
        Input: Grayscale image (H, W) normalized 0-255
        Output: High-pass filtered DCT coefficient map
        """
        h, w = image_gray.shape
        # Ensure image is divisible by block size
        h_pad = (self.block_size - h % self.block_size) % self.block_size
        w_pad = (self.block_size - w % self.block_size) % self.block_size
        image_padded = np.pad(image_gray, ((0, h_pad), (0, w_pad)), mode='edge')
        
        dct_map = np.zeros_like(image_padded, dtype=np.float32)
        
        # Process block by block
        for i in range(0, h, self.block_size):
            for j in range(0, w, self.block_size):
                block = image_padded[i:i+self.block_size, j:j+self.block_size]
                
                # 1. Apply DCT (Type II)
                dct_block = scipy.fftpack.dct(
                    scipy.fftpack.dct(block.T, norm='ortho').T, norm='ortho'
                )
                
                # 2. Zero out low frequencies (Top-Left Quadrant)
                # The paper suggests masking the DC coefficient and low-freqs.
                # A simple High-Pass filter: mask the 4x4 top-left corner
                mask = np.ones((self.block_size, self.block_size))
                mask[0:4, 0:4] = 0 
                
                dct_filtered = dct_block * mask
                
                # 3. Store result (No IDCT needed, we feed coefficients to CNN)
                dct_map[i:i+self.block_size, j:j+self.block_size] = dct_filtered
                
        return dct_map[:h, :w]

# --- PART 2: The Architecture (Dual Stream) ---
class FrequencyStream(nn.Module):
    """
    Lightweight CNN to process the DCT maps.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        # Placeholder linear layer; dynamic sizing needed for production
        self.fc = nn.LazyLinear(512) 

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class RGBStream(nn.Module):
    """
    Lightweight CNN backbone for demo purposes (replaces EfficientNet-B4).
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.LazyLinear(1792) # Match original output dim for compatibility

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class DualStreamNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_stream = RGBStream()
        self.freq_stream = FrequencyStream()
        
        # Fusion Layer
        self.fusion = nn.LazyLinear(128)
        self.classifier = nn.Linear(128, 2) # Real vs Fake

    def forward(self, x_rgb, x_freq):
        feat_rgb = self.rgb_stream(x_rgb)
        feat_freq = self.freq_stream(x_freq)
        
        # Concatenate
        combined = torch.cat((feat_rgb, feat_freq), dim=1)
        
        # Fusion
        embedding = F.relu(self.fusion(combined))
        
        # Classification
        out = self.classifier(embedding)
        return out, embedding 

# --- PART 3: Proof of Concept Simulation ---
def run_demo():
    print("Initializing Research Pipeline...")
    
    # 1. Create Data
    try:
        real_img = data.astronaut()
        if len(real_img.shape) == 3:
            real_img = cv2.cvtColor(real_img, cv2.COLOR_RGB2GRAY)
        real_img = cv2.resize(real_img, (224, 224))
    except Exception as e:
        print(f"Fallback to noise: {e}")
        real_img = np.random.randint(0, 255, (224, 224), dtype=np.uint8)

    # 2. Create a "Fake" version (Upsampling Artifacts)
    h, w = real_img.shape
    # Extreme downsample to force upsampler to invent data (hallucinate)
    small = cv2.resize(real_img, (h//4, w//4), interpolation=cv2.INTER_NEAREST)
    fake_img = cv2.resize(small, (h, w), interpolation=cv2.INTER_CUBIC)

    # 3. Apply DCT Processing
    processor = DCTProcessor()
    
    dct_real = processor.process_image(real_img)
    dct_fake = processor.process_image(fake_img)
    
    # 4. Visualization
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 2, 1)
    plt.title("Real Image (Spatial)")
    plt.imshow(real_img, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title("Fake Image (Upsampled)")
    plt.imshow(fake_img, cmap='gray')
    plt.axis('off')

    # Log scale for better visibility of artifacts
    # Adding 1e-6 to avoid log(0)
    plt.subplot(2, 2, 3)
    plt.title("Real: DCT High-Freq (Log Scale)")
    plt.imshow(np.log(np.abs(dct_real) + 1e-6), cmap='inferno')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title("Fake: DCT High-Freq (Log Scale)")
    plt.imshow(np.log(np.abs(dct_fake) + 1e-6), cmap='inferno')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('dct_proof.png')
    print("Proof generated: 'dct_proof.png'")
    print("ANALYSIS: Compare the bottom-right vs bottom-left images.")
    print("The 'Fake' DCT map will show distinct grid patterns (checkerboarding) or aliasing artifacts")
    print("introduced by the bicubic upsampler. This is what the Frequency Stream learns.")

if __name__ == "__main__":
    run_demo()
