import numpy as np
import scipy.fftpack
import cv2

class DCTProcessor:
    """
    Implements the Artifact-Invariant DCT Feature Extraction.
    Ref: Section III.A of the paper.
    """
    def __init__(self, block_size=8):
        self.block_size = block_size

    def process_image(self, image):
        """
        Input: Grayscale image (H, W) or (H, W, C)
        Output: High-pass filtered DCT coefficient map normalized to [0, 1]
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
        h, w = image.shape
        # Pad to multiples of block_size
        h_pad = (self.block_size - h % self.block_size) % self.block_size
        w_pad = (self.block_size - w % self.block_size) % self.block_size
        image_padded = np.pad(image, ((0, h_pad), (0, w_pad)), mode='edge')
        
        dct_map = np.zeros_like(image_padded, dtype=np.float32)
        
        # Block-wise DCT
        for i in range(0, h, self.block_size):
            for j in range(0, w, self.block_size):
                block = image_padded[i:i+self.block_size, j:j+self.block_size]
                
                # Type II DCT
                dct_block = scipy.fftpack.dct(
                    scipy.fftpack.dct(block.T, norm='ortho').T, norm='ortho'
                )
                
                # High-Pass Filter: Zero out top-left 4x4 frequencies
                mask = np.ones((self.block_size, self.block_size))
                mask[0:4, 0:4] = 0 
                
                dct_filtered = dct_block * mask
                dct_map[i:i+self.block_size, j:j+self.block_size] = dct_filtered
                
        # Crop back to original size
        dct_map = dct_map[:h, :w]
        
        # Log-Normalization (Crucial for Neural Network stability)
        dct_map = np.log(np.abs(dct_map) + 1e-6)
        
        # Instance Min-Max Normalization
        d_min, d_max = dct_map.min(), dct_map.max()
        if d_max - d_min > 0:
            dct_map = (dct_map - d_min) / (d_max - d_min)
        else:
            dct_map = np.zeros_like(dct_map)
            
        return dct_map
