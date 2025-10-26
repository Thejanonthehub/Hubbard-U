import torch

class Predictor:
    def __init__(self, mlp_model, rf_model, data_module, alpha=0.7):
        self.model = mlp_model
        self.rf_model = rf_model
        self.scaler = data_module.scaler
        self.poly = data_module.poly
        self.alpha = alpha
    
    def predict(self, X_input, raw=True):
        if raw:
            X_poly = self.poly.transform(X_input)
            X_scaled = self.scaler.transform(X_poly)
        else:
            X_scaled = X_input
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            y_pred_mlp = self.model(X_tensor).numpy()
        y_pred_rf = self.rf_model.predict(X_scaled)
        y_pred_ens = self.alpha * y_pred_rf + (1 - self.alpha) * y_pred_mlp
        return y_pred_mlp, y_pred_rf, y_pred_ens
