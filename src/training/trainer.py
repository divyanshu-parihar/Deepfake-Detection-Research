import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from sklearn.metrics import accuracy_score, roc_auc_score
from src.core.networks import DualStreamNetwork
from src.data.dataset import SyntheticDeepfakeDataset

class DeepfakeTrainer:
    def __init__(self, batch_size=32, lr=0.0005, epochs=5, device=None):
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Trainer initialized on {self.device}")
        
        # Initialize Data
        self.train_set = SyntheticDeepfakeDataset(mode='train')
        self.test_set = SyntheticDeepfakeDataset(mode='test')
        
        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_set, batch_size=batch_size, shuffle=False)
        
        # Initialize Model
        self.model = DualStreamNetwork(mode='lightweight').to(self.device)
        
        # Initialize Optimization
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Setup Checkpoints
        if not os.path.exists('models'):
            os.makedirs('models')

    def train(self):
        print(f"\n=== Starting Training for {self.epochs} Epochs ===")
        best_acc = 0.0
        
        # Initialize Lazy Modules with a dummy pass
        dummy_rgb = torch.randn(1, 3, 224, 224).to(self.device)
        dummy_freq = torch.randn(1, 1, 224, 224).to(self.device)
        self.model(dummy_rgb, dummy_freq)
        
        for epoch in range(self.epochs):
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
            print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")
            
            # Evaluate
            test_acc, auc = self.evaluate()
            print(f"          Test Acc:  {test_acc:.2f}% | AUC: {auc:.4f}")
            
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(self.model.state_dict(), "models/best_deepfake_detector.pth")
                print("          >>> NEW RECORD! Model saved.")
                
        print(f"\nFinal Best Accuracy: {best_acc:.2f}%")
        
    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.test_loader:
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
