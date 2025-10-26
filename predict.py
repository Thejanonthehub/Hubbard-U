import os
import pandas as pd
import numpy as np
import torch
from pymatgen.core import Structure, Element
from pymatgen.analysis.local_env import CrystalNN
import joblib

# -----------------------------
# Feature extraction class
# -----------------------------
class Fe_for_pred:
    def __init__(self, cif_file, output_csv, target_species, flush_interval=1):
        self.cif_file = cif_file
        self.output_csv = output_csv
        self.target_species = target_species
        self.flush_interval = flush_interval
        self.features_list = []

    def get_atomic_properties(self, species):
        el = Element(species)
        return {
            "electronegativity": el.X,
            "ionization_energy": el.ionization_energy,
            "electron_affinity": el.electron_affinity,
        }

    def extract_features(self):
        """Extract features from a single CIF file."""
        if not os.path.exists(self.cif_file):
            raise FileNotFoundError(f"No CIF file found at {self.cif_file}")

        structure = Structure.from_file(self.cif_file)

        species = self.target_species
        atomic_props = self.get_atomic_properties(species)

        # Neighbor analysis using CrystalNN
        try:
            cnn = CrystalNN()
            avg_nn_dist = np.mean([np.mean([np.linalg.norm(n['site'].coords - structure[i].coords)
                                           for n in cnn.get_nn_info(structure, i)])
                                   for i, site in enumerate(structure.sites)
                                   if site.specie.symbol == species])
        except:
            avg_nn_dist = None

        # Sum of neighbor atomic numbers
        try:
            s_stm_v = 0
            for i, site in enumerate(structure.sites):
                if site.specie.symbol == species:
                    neighbors = cnn.get_nn_info(structure, i)
                    s_stm_v += sum(nei['site'].specie.number for nei in neighbors)
        except:
            s_stm_v = None

        feature_dict = {
            "Species": species,
            "avg_nn_dist": avg_nn_dist,
            "sum_atn_nn": s_stm_v,
            "electronegativity": atomic_props["electronegativity"],
            "ionization_energy": atomic_props["ionization_energy"],
            "electron_affinity": atomic_props["electron_affinity"],
        }

        self.features_list.append(feature_dict)
        df_features = pd.DataFrame(self.features_list)
        df_features.to_csv(self.output_csv, index=False)

        # Return features as numpy array in correct order
        return np.array([avg_nn_dist, s_stm_v,
                         atomic_props["electronegativity"],
                         atomic_props["ionization_energy"],
                         atomic_props["electron_affinity"]]).reshape(1, -1)

# -----------------------------
# Load saved models
# -----------------------------
def load_models(prefix='hubbard_uj'):
    poly = joblib.load(f'{prefix}_poly.pkl')
    scaler = joblib.load(f'{prefix}_scaler.pkl')
    rf_model = joblib.load(f'{prefix}_rf.pkl')

    class MLP(torch.nn.Module):
        def __init__(self, input_size, hidden_size=128, output_size=2, dropout=0.2):
            super().__init__()
            self.fc1 = torch.nn.Linear(input_size, hidden_size)
            self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
            self.fc3 = torch.nn.Linear(hidden_size, hidden_size)
            self.fc4 = torch.nn.Linear(hidden_size, output_size)
            self.relu = torch.nn.ReLU()
            self.dropout = torch.nn.Dropout(dropout)
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.relu(self.fc3(x))
            x = self.fc4(x)
            return x

    input_size = poly.transform(np.zeros((1,5))).shape[1]
    mlp_model = MLP(input_size=input_size)
    mlp_model.load_state_dict(torch.load(f'{prefix}_mlp.pth'))
    mlp_model.eval()
    return poly, scaler, rf_model, mlp_model

# -----------------------------
# Predict values
# -----------------------------
def predict_values(X_input, alpha=0.7):
    poly, scaler, rf_model, mlp_model = load_models()
    X_poly = poly.transform(X_input)
    X_scaled = scaler.transform(X_poly)
    
    y_rf = rf_model.predict(X_scaled)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        y_mlp = mlp_model(X_tensor).numpy()
    y_ens = alpha * y_rf + (1 - alpha) * y_mlp
    return y_mlp, y_rf, y_ens

# -----------------------------
# Main script
# -----------------------------
if __name__ == "__main__":
    cif_file = input("Enter path to CIF file: ").strip()
    target_species = input("Enter species to analyze (e.g., Ag, C, H): ").strip()

    fe = Fe_for_pred(cif_file=cif_file,
                     output_csv="predict_features.csv",
                     target_species=target_species)
    X_input = fe.extract_features()

    mlp_pred, rf_pred, ens_pred = predict_values(X_input)

    print("\nPredicted Properties:")
    print(f"MLP Prediction (U, J): {mlp_pred[0]}")
    print(f"RF Prediction (U, J): {rf_pred[0]}")
    print(f"Ensemble Prediction (U, J): {ens_pred[0]}")
