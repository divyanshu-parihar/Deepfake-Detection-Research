import matplotlib.pyplot as plt
import numpy as np
import cv2
from src.core.dct import DCTProcessor
from skimage import data

def generate_proof_image(save_path='assets/dct_proof.png'):
    """
    Generates a visual comparison between a Real image and a simulated Fake image
    in both the Spatial and Frequency (DCT) domains.
    """
    print("Generating Proof-of-Concept Visualization...")
    
    # 1. Get Source Image
    try:
        real_img = data.astronaut()
        if len(real_img.shape) == 3:
            real_img = cv2.cvtColor(real_img, cv2.COLOR_RGB2GRAY)
        real_img = cv2.resize(real_img, (224, 224))
    except:
        real_img = np.random.randint(0, 255, (224, 224), dtype=np.uint8)

    # 2. Create "Fake" (Upsampled)
    h, w = real_img.shape
    # Downsample then Upsample (creates spectral aliases)
    small = cv2.resize(real_img, (h//4, w//4), interpolation=cv2.INTER_NEAREST)
    fake_img = cv2.resize(small, (h, w), interpolation=cv2.INTER_CUBIC)

    # 3. Process DCT
    processor = DCTProcessor()
    dct_real = processor.process_image(real_img)
    dct_fake = processor.process_image(fake_img)
    
    # 4. Plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Spatial Real
    axes[0, 0].imshow(real_img, cmap='gray')
    axes[0, 0].set_title("Real Image (Spatial)")
    axes[0, 0].axis('off')

    # Spatial Fake
    axes[0, 1].imshow(fake_img, cmap='gray')
    axes[0, 1].set_title("Fake Image (Upsampled)")
    axes[0, 1].axis('off')

    # Freq Real
    im1 = axes[1, 0].imshow(dct_real, cmap='inferno')
    axes[1, 0].set_title("Real: DCT Residuals")
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Freq Fake
    im2 = axes[1, 1].imshow(dct_fake, cmap='inferno')
    axes[1, 1].set_title("Fake: DCT Residuals (Artifacts Visible)")
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Proof saved to {save_path}")
