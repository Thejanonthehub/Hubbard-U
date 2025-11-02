# ⚡ UPredictor v1.0.0

![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Build](https://img.shields.io/badge/Build-Passing-brightgreen)
![Coverage](https://img.shields.io/badge/Coverage-95%25-yellow)
![Version](https://img.shields.io/badge/Version-1.0.0-orange)


**UPredictor** is a command-line tool that predicts **Hubbard U and J values** for custom materials using a pre-trained ensemble model (MLP + RF).  
It only requires a `.cif` file and the target atomic species — perfect for rapid screening before DFT+U calculations.  

---

## 🌍 Overview  

The **Hubbard U parameter** plays a crucial role in correcting electron correlation effects in DFT+U calculations, especially for materials with localized *d* and *f* orbitals.  
However, obtaining accurate **U values** is **computationally expensive** — often requiring complex linear-response or cDFT calculations.  

➡️ This work employs a **hybrid MLP–Random Forest ensemble model** to **predict Hubbard U** values for **custom materials**.  
The **Multilayer Perceptron (MLP)** component is a feedforward **neural network** that captures complex nonlinear relationships, while the **Random Forest (RF)** provides robust decision boundaries and interpretable feature importance.  
Predictions are obtained via a **weighted blending** of MLP and RF outputs, trained on data from the **Materials Project**.

---

## ⚙️ Workflow  

### 1. **Data Source** 🧩
- Dataset obtained from the [Materials Project](https://materialsproject.org/)
- Includes chemical compositions, orbital descriptors, and reference Hubbard U values

### 2. **Feature Engineering** 🧱
- Extract structural and electronic descriptors using `pymatgen` and `matminer`
- Apply polynomial feature expansion and data scaling for enhanced representation
- Prepare feature matrices suitable for both neural and tree-based models

### 3. **Model Architecture** 🧬
- **Hybrid MLP–Random Forest Ensemble**
  - **Multilayer Perceptron (MLP):** a feedforward neural network with:
    - Multiple fully connected layers (`nn.Linear` in PyTorch)
    - Nonlinear activation functions (ReLU)
    - Dropout for regularization
    - Learns complex nonlinear relationships between input features
  - **Random Forest (RF):** provides robust decision boundaries and interpretable feature importance
- Final prediction obtained via a **weighted blending** of MLP and RF outputs

### 4. **Training** 🔥
- Frameworks:
  - `PyTorch` for the MLP component (neural network)
  - `scikit-learn` for the Random Forest
- Loss Metric: **Mean Absolute Error (MAE)**
- Optimizer: **AdamW** (for the MLP)
- Model evaluation via **5-fold cross-validation** to ensure robust performance and generalization

### 5. **Prediction** 🚀
- Input material features (optionally derived from `.cif` files)
- The hybrid ensemble predicts the **Hubbard U** parameter within seconds
- Combines the strengths of both neural and ensemble learning methods for improved accuracy and reliability

---

## 🚀 Quick Start  

### 🧩 1. Clone & Setup  
```bash
    git clone https://github.com/yourusername/OCGNN-HubbardU.git
    cd OCGNN-HubbardU
    python3 -m venv jlab_env
    source jlab_env/bin/activate      # macOS/Linux
    jlab_env\Scripts\activate       # Windows

    pip install -r requirements.txt 
```

###  ⚙️ 2. Install the CLI Tool
**From inside the project directory, run:**

```bash 
    pip install -e . 
```

    This registers the command-line interface upredictor, allowing you to run it anywhere.

### 🧠 3. Run the Predictor

    Simply type:  ```bash upredictor``` 


**You’ll see the startup banner:**
```bash
       __  ______                ___      __            
      / / / / __ \________  ____/ (_)____/ /_____  _____
     / / / / /_/ / ___/ _ \/ __  / / ___/ __/ __ \/ ___/
    / /_/ / ____/ /  /  __/ /_/ / / /__/ /_/ /_/ / /    
    \____/_/   /_/   \___/\__,_/_/\___/\__/\____/_/     

                                                    
    Welcome to the UPredictor ML Engine!
    ------------------------------------
```
**Then follow the interactive prompts:**
```bash
    Enter path to CIF file: /path/to/material.cif
    Enter species to analyze (e.g., Fe, C, Ni): Fe 
```

**After a few seconds, it outputs predictions:**
```bash
    Predicted Properties:
    MLP Prediction (U, J): [4.32, 0.71]
    RF Prediction (U, J): [4.28, 0.68]
    Ensemble Prediction (U, J): [4.30, 0.70] 
```

### ⚠️ 4. Troubleshooting
**Error:**
```bah
    FileNotFoundError: './engine/hubbard_uj_poly.pkl' 
```

**Solution:**
    Ensure the engine/ folder exists in the same directory where you run the command and includes:
```bash
    engine/
     ├── hubbard_uj_poly.pkl
     ├── hubbard_uj_scaler.pkl
     ├── hubbard_uj_rf.pkl
     └── hubbard_uj_mlp.pth
 ```

### 💡 5. Tip

        UPredictor works offline after setup and supports any CIF file compatible with pymatgen.
    
### 🧩 6. Example Workflow

        Prepare your custom .cif file.
        Run upredictor from the terminal.
        Enter:
        The path to your CIF file
        The target species (e.g., Fe, C, Ni)
        Get instant predictions for Hubbard U and J.

