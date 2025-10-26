import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class Evaluator:
    def __init__(self, predictor, data_module):
        self.predictor = predictor
        self.X_test = data_module.X_test
        self.y_test = data_module.y_test
    
    def evaluate(self):
        y_pred_mlp, y_pred_rf, y_pred_ens = self.predictor.predict(self.X_test, raw=False)
        y_true = self.y_test
        
        results = {
            'MLP_U_RMSE': np.sqrt(mean_squared_error(y_true[:,0], y_pred_mlp[:,0])),
            'MLP_J_RMSE': np.sqrt(mean_squared_error(y_true[:,1], y_pred_mlp[:,1])),
            'MLP_U_R2': r2_score(y_true[:,0], y_pred_mlp[:,0]),
            'MLP_J_R2': r2_score(y_true[:,1], y_pred_mlp[:,1]),
            'Ensemble_U_RMSE': np.sqrt(mean_squared_error(y_true[:,0], y_pred_ens[:,0])),
            'Ensemble_J_RMSE': np.sqrt(mean_squared_error(y_true[:,1], y_pred_ens[:,1])),
            'Ensemble_U_R2': r2_score(y_true[:,0], y_pred_ens[:,0]),
            'Ensemble_J_R2': r2_score(y_true[:,1], y_pred_ens[:,1]),
        }
        return results
