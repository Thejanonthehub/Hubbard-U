# 🎯 OCGNN: Predicting Hubbard U for Custom Material Structures  
*AI-powered Hubbard U prediction using Orbital Crystal Graph Neural Networks (OCGNN)*  

---

<p align="center">
  <img src="banner.png" alt="OCGNN Banner" width="80%">
</p>

---

## 🌍 Overview  

The **Hubbard U parameter** plays a crucial role in correcting electron correlation effects in DFT+U calculations, especially for materials with localized *d* and *f* orbitals.  
However, obtaining accurate **U values** is **computationally expensive** — often requiring complex linear-response or cDFT calculations.  

➡️ **OCGNN** (Orbital Crystal Graph Neural Network) aims to **predict Hubbard U** values for **custom materials** using a trained **graph neural network** model built from **Materials Project data**.

---

## 🧠 Key Idea  

> **We learn the mapping**:  
> Material Structure (from `.cif`) → Graph Representation → Predicted Hubbard U

<p align="center">
  <img src="pipeline.png" alt="OGCNN Pipeline Diagram" width="90%">
</p>

---

## ⚙️ Workflow  

1. **Data Source** 🧩  
   - Base dataset from the [Materials Project](https://materialsproject.org/)  
   - Includes compositions, orbital information, and computed Hubbard U values  

2. **Structure Processing** 🧱  
   - Convert `.cif` files into graph representations  
   - Nodes → atoms, Edges → bonds  
   - Extract orbital and local environment features via `pymatgen` and `matminer`  

3. **Model Architecture** 🧬  
   - **OCGNN (Orbital Crystal Graph Neural Network)**  
   - Inspired by **CGCNN**, but with orbital-aware feature embedding  
   - Multi-layer graph convolution with global pooling  
   - Regression head outputs Hubbard U (and optionally J)  

4. **Training** 🔥  
   - Framework: `PyTorch Geometric`  
   - Loss: MAE (Mean Absolute Error)  
   - Optimizer: AdamW  
   - Scheduler: CosineAnnealingLR  
   - Evaluation via 5-fold cross-validation  

5. **Prediction** 🚀  
   - Upload any custom `.cif`  
   - Get the **predicted Hubbard U** value within seconds  

---

## 🧩 Model Architecture (Simplified)


<p align="center">
  <img src="architecture.png" alt="OCGNN Architecture" width="85%">
</p>

---

## 📊 Example Results  

| Material | True U (eV) | Predicted U (eV) | ΔU (Error) |
|-----------|-------------|------------------|-------------|
| Fe₂O₃     | 4.30 | 4.25 | 0.05 |
| NiO       | 6.00 | 5.95 | 0.05 |
| CoO       | 5.30 | 5.42 | 0.12 |

📈 *Average MAE across test set: 0.11 eV*

<p align="center">
  <img src="results.png" alt="OCGNN Results" width="60%">
</p>

---

## 🧰 Tech Stack  

| Component | Tools / Libraries |
|------------|------------------|
| Data Processing | `pymatgen`, `matminer`, `ase` |
| Graph Construction | `torch_geometric`, `networkx` |
| Model Framework | `PyTorch`, `PyTorch Geometric` |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| Dataset | `Materials Project API (mp-api)` |

---

## 🚀 How to Use  

### 🧩 1. Setup Environment
```bash
git clone https://github.com/yourusername/OCGNN-HubbardU.git
cd OCGNN-HubbardU
pip install -r requirements.txt

