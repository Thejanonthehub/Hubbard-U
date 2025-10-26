# training.py
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

class Trainer:
    def __init__(self, model, train_loader, lr=0.001):
        self.model = model
        self.train_loader = train_loader
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=500)

    def train(self, num_epochs=500, patience=50):
        best_loss = float('inf')
        counter = 0
        self.model.train()

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            self.scheduler.step()

            avg_loss = epoch_loss / len(self.train_loader)
            if avg_loss < best_loss:
                best_loss = avg_loss
                counter = 0
                torch.save(self.model.state_dict(), 'best_mlp.pth')
            else:
                counter += 1

            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")

            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        self.model.load_state_dict(torch.load('best_mlp.pth'))
        print("✅ MLP training complete.")
