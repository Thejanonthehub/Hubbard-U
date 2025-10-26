# dataset.py
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

class HubbardDataset:
    def __init__(self, u_j_file, features_file, target_cols=None, base_feature_cols=None, train_size=0.8):
        self.u_j_df = pd.read_csv(u_j_file)
        self.features_df = pd.read_csv(features_file)
        self.target_cols = target_cols or ['U(eV)', 'J(eV)']
        self.base_feature_cols = base_feature_cols or [
            'avg_nn_dist', 'sum_atn_nn',
            'electronegativity', 'ionization_energy', 'electron_affinity'
        ]
        self.train_size = train_size

        self.poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        self.scaler = StandardScaler()

        self.prepare_data()

    def prepare_data(self):
        merged = self.u_j_df.merge(self.features_df, on='identifier')
        X_raw = merged[self.base_feature_cols].fillna(0).values
        y = merged[self.target_cols].fillna(0).values

        X_poly = self.poly.fit_transform(X_raw)
        X_scaled = self.scaler.fit_transform(X_poly)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, train_size=self.train_size, random_state=42
        )

        self.X_train_tensor = torch.tensor(self.X_train, dtype=torch.float32)
        self.y_train_tensor = torch.tensor(self.y_train, dtype=torch.float32)
        self.X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32)
        self.y_test_tensor = torch.tensor(self.y_test, dtype=torch.float32)

        self.train_dataset = TensorDataset(self.X_train_tensor, self.y_train_tensor)
        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
