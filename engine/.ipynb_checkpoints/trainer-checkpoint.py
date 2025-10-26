import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

class Trainer:
    def __init__(self, mlp_model, rf_model, data_module):
        self.model = mlp_model
        self.rf_model = rf_model
        self.train_loader = data_module.train_loader
        self.X_train = data_module.X_train
        self.y_train = data_module.y_train
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
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
                print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}")
            
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best MLP model
        self.model.load_state_dict(torch.load('best_mlp.pth'))
        # Train RF
        self.rf_model.fit(self.X_train, self.y_train)
        print(f"RF OOB Score: {self.rf_model.model.oob_score_:.3f}")
