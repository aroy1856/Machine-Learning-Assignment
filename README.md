# Trip-Level Purchase Conversion Predictor

This project predicts whether a shopper’s trip will convert into a purchase using an XGBoost classification model.  
It includes:
- An **EDA + experiment notebook** for detailed data exploration.
- A **Streamlit app** for interactive predictions and retraining.

---

## Project Structure

```
.
├── app.py                # Streamlit app entry point
├── preprocessing.py      # Data preprocessing functions
├── model.py              # Model training & evaluation functions
├── model/                # Saved XGBoost model & Scaler
│   ├── xgb_model.pkl
│   └── scaler.pkl
├── data/
│   └── trip_dataset.csv  # Raw input dataset
├── notebook/
│   └── Trip-Level Purchase.ipynb  # EDA & experimentation
├── requirement.txt       # Python dependencies
├── README.md             # Project overview
└── __pycache__/          # Python cache files
```

---

## How it works

**Notebook:**  
Use `notebook/Trip-Level Purchase.ipynb` to:
- Explore the dataset.
- Perform EDA.
- Build, compare and tune models.
- Finalize the XGBoost pipeline.

**Streamlit App:**  
`app.py` lets you:
- Load the trained XGBoost model (or retrain it on demand).
- Upload or interact with new trip data.
- Predict if a trip will convert.
- Visualize feature importance using SHAP.

---

## Run the App

```
bash
# Create a conda environment (optional)
conda create -n trip_env python=3.10
conda activate trip_env

# Install dependencies
pip install -r requirement.txt

# Run the Streamlit app
streamlit run app.py
```

---

## Key Features

- **Automatic Model Reloading:**  
  If `model/xgb_model.pkl` exists, the app loads it. Otherwise, it trains a new one.

- **Scaler Consistency:**  
  Input data is scaled using the same `scaler.pkl` saved during training.

- **Manual Prediction:**  
  Use interactive widgets to test new trip scenarios.

- **Feature Importance:**  
  SHAP plots explain which features drive the prediction.

---

## Data

- **Input:** `data/trip_dataset.csv`  
  Contains raw event-level data with views, add-to-cart actions, purchases, promos, prices, device, and region info.

---

## Requirements

- Python 3.10+
- Key packages: `streamlit`, `pandas`, `xgboost`, `shap`, `scikit-learn`, `matplotlib`.

---

## Next Steps

- Deploy on Streamlit Cloud or similar.

---

## Author

Maintained by Abhishek Roy