# evaluation.py
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class Evaluator:
    def __init__(self, mlp_model, rf_model, poly, scaler, alpha=0.7):
        self.mlp_model = mlp_model
        self.rf_model = rf_model
        self.poly = poly
        self.scaler = scaler
        self.alpha = alpha

    def predict(self, X_input, raw=True):
        if raw:
            X_poly = self.poly.transform(X_input)
            X_scaled = self.scaler.transform(X_poly)
        else:
            X_scaled = X_input

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        self.mlp_model.eval()
        with torch.no_grad():
            y_pred_mlp = self.mlp_model(X_tensor).numpy()

        y_pred_rf = self.rf_model.predict(X_scaled)
        y_pred_ens = self.alpha * y_pred_rf + (1 - self.alpha) * y_pred_mlp
        return y_pred_mlp, y_pred_rf, y_pred_ens

    def evaluate(self, X_test, y_test):
        y_pred_mlp, _, y_pred_ens = self.predict(X_test, raw=False)
        y_true = y_test

        results = {
            'MLP_U_RMSE': np.sqrt(mean_squared_error(y_true[:, 0], y_pred_mlp[:, 0])),
            'MLP_J_RMSE': np.sqrt(mean_squared_error(y_true[:, 1], y_pred_mlp[:, 1])),
            'MLP_U_R2': r2_score(y_true[:, 0], y_pred_mlp[:, 0]),
            'MLP_J_R2': r2_score(y_true[:, 1], y_pred_mlp[:, 1]),
            'ENS_U_RMSE': np.sqrt(mean_squared_error(y_true[:, 0], y_pred_ens[:, 0])),
            'ENS_J_RMSE': np.sqrt(mean_squared_error(y_true[:, 1], y_pred_ens[:, 1])),
            'ENS_U_R2': r2_score(y_true[:, 0], y_pred_ens[:, 0]),
            'ENS_J_R2': r2_score(y_true[:, 1], y_pred_ens[:, 1]),
        }

        print("✅ Evaluation Results:")
        for k, v in results.items():
            print(f"{k}: {v:.4f}")
        return results
