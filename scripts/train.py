import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.trainer import DeepfakeTrainer

if __name__ == "__main__":
    trainer = DeepfakeTrainer(epochs=5)
    trainer.train()
