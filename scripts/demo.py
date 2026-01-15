import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.visualization import generate_proof_image

if __name__ == "__main__":
    generate_proof_image()
