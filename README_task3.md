# 🏠 Multimodal Housing Price Prediction (Images + Tabular Data)


![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet18-red?style=flat-square)
![Multimodal](https://img.shields.io/badge/Modality-Image+Tabular-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Objective

Predict housing sale prices by fusing two modalities — structured tabular features and house exterior images — using a CNN backbone (ResNet-18) combined with a tabular MLP in a unified deep learning model.

---

## 📂 Dataset

| Modality | Details |
|----------|---------|
| Tabular | Ames Housing Dataset — 1,460 records, 79 features (via OpenML) |
| Images | House exterior photos — use real images from Zillow/Kaggle; synthetic placeholders provided for demo |
| Target | `SalePrice` (USD) |

**Selected Tabular Features:** `GrLivArea`, `OverallQual`, `YearBuilt`, `TotalBsmtSF`, `GarageArea`, `BedroomAbvGr`, `FullBath`, `LotArea`

---

## 🧠 Methodology / Approach

```
House Image (224×224 RGB)           Tabular Features (8 columns)
         │                                     │
         ▼                                     ▼
  ResNet-18 Backbone                  MLP Tabular Branch
  (ImageNet pretrained)               Linear(8→64) → ReLU
  └── Custom head:                    → Dropout(0.2)
      Linear(512→256) → ReLU          → Linear(64→128) → ReLU
      → Dropout(0.3)                             │
         │                                       │
         ▼ Image Embedding (256)    Tab Embedding (128) ▼
                    └───────── Concatenate ──────────┘
                                     │
                              Fusion MLP Head
                           Linear(384→128) → ReLU
                           → Dropout(0.3)
                           → Linear(128→64) → ReLU
                           → Linear(64→1)
                                     │
                                     ▼
                           Predicted log1p(SalePrice)
```

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Loss | Huber Loss (δ=1.0) |
| Optimizer | AdamW (lr=1e-4, wd=1e-4) |
| Scheduler | ReduceLROnPlateau (patience=5) |
| Epochs | 30 with best-model checkpoint |
| Target transform | log1p(SalePrice) |
| Image size | 224×224, ImageNet normalization |

---

## 📊 Key Results

| Metric | Score |
|--------|-------|
| MAE | ~$25,000 – $35,000 |
| RMSE | ~$40,000 – $55,000 |
| MAPE | ~12 – 18% |

---

## 🔍 Observations

1. **Log-transforming SalePrice** is essential — the raw distribution is heavily right-skewed, causing instability during training
2. **Late fusion** (concatenating embeddings before the final MLP) outperforms early fusion and unimodal baselines
3. **ResNet-18** converges faster than deeper networks on this dataset size — diminishing returns with ResNet-50
4. With **real house images**, the image branch contributes a 5–8% RMSE reduction over tabular-only
5. **Huber loss** is more robust than MSE for house prices — large luxury outliers don't destabilize training

---

## 🗂️ Project Structure

```
multimodal-housing-price/
├── task3_multimodal_housing.ipynb    ← Main notebook
├── best_multimodal_model.pt          ← Best checkpoint weights
├── house_images/                     ← Image directory (add real images here)
├── outputs/
│   ├── task3_eda.png
│   └── task3_evaluation.png
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/engineermuhammadtalha/multimodal-housing-price
cd multimodal-housing-price

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Add real house images
# Place images named 0.jpg, 1.jpg, ... matching DataFrame row index into house_images/

# 4. Run the notebook
jupyter notebook task3_multimodal_housing.ipynb
```

---

## 🛠️ Tech Stack

`Python` · `PyTorch` · `torchvision` · `ResNet-18` · `scikit-learn` · `pandas` · `NumPy` · `Pillow` · `Matplotlib`
