import torch
import joblib

class Saver:
    def __init__(self, mlp_model, rf_model, data_module):
        self.model = mlp_model
        self.rf_model = rf_model
        self.scaler = data_module.scaler
        self.poly = data_module.poly
    
    def save(self, prefix='hubbard_uj'):
        torch.save(self.model.state_dict(), f'{prefix}_mlp.pth')
        joblib.dump(self.rf_model.model, f'{prefix}_rf.pkl')
        joblib.dump(self.scaler, f'{prefix}_scaler.pkl')
        joblib.dump(self.poly, f'{prefix}_poly.pkl')
        print("✅ Models and scalers saved.")
