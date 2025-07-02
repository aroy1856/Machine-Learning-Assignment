import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
import os
import pickle

from preprocessing import preprocess_raw_data, train_test_preprocess
from model import train_xgboost, evaluate_model

st.set_page_config(page_title="Trip Conversion Predictor", layout="wide")

MODEL_PATH = "xgb_model.pkl"

@st.cache_data
def load_data():
    df = pd.read_csv("data/trip_dataset.csv", parse_dates=['event_time'])
    return df

df = load_data() 
X, y = preprocess_raw_data(df)
X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = train_test_preprocess(X, y)

@st.cache_resource
def get_trained_model():
    if os.path.exists(MODEL_PATH):
        model = pickle.load(MODEL_PATH)
        st.success("✅ Loaded model from file.")
    else:
        model = train_xgboost(X_train_scaled, y_train)
        st.success("✅ Trained new model and saved to file.")
    return model

# 👉 Add retrain option
if st.sidebar.button("🔄 Retrain Model"):
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)  # Delete old model file
    st.cache_resource.clear()  # Clear Streamlit's cached model
    st.success("✅ Model retrained!")

model = get_trained_model()

train_metrics = evaluate_model(model, X_train_scaled, y_train)
test_metrics = evaluate_model(model, X_test_scaled, y_test)

st.title("🛒 XGBoost Trip Conversion Predictor")

st.subheader("Model Performance (Test Set)")
st.write(f"""
- **Accuracy:** {test_metrics['accuracy']:.4f}  
- **F1 Score:** {test_metrics['f1']:.4f}  
- **Precision:** {test_metrics['precision']:.4f}  
- **Recall:** {test_metrics['recall']:.4f}  
- **ROC AUC:** {test_metrics['roc_auc']:.4f}  
""")


def get_explainer(_model):
    return shap.Explainer(_model)

explainer = get_explainer(model)

def get_shap_values(_explainer, _X_train_scaled):
    return _explainer(_X_train_scaled)

shap_values = get_shap_values(explainer, X_train_scaled)


# Manual input
st.subheader("🎯 Predict a New Trip")

total_views = st.number_input("Total Views", min_value=0, step=1)
total_add_to_cart = st.number_input("Total Add to Cart", min_value=0, step=1)
promo_interactions = st.number_input("Promo Interactions", min_value=0, step=1)
promo_rate = st.slider("Promo Rate", min_value=0.0, max_value=1.0, step=0.01)
avg_price = st.number_input("Average Product Price", min_value=0.0, step=0.1)
dominant_device = st.selectbox("Dominant Device Type", options=["mobile", "tablet", "desktop"])
region = st.selectbox("Region", options=["North", "South", "East", "West"])

view_to_cart_ratio = total_add_to_cart / (total_views + 1e-6)

input_dict = {
    "event_type_n_views": total_views,
    "event_type_n_add_to_cart": total_add_to_cart,
    "is_promo_n_promo_interactions": promo_interactions,
    "is_promo_promo_interaction_rate": promo_rate,
    "price_avg_price_viewed": avg_price,
    "view_to_cart_ratio": view_to_cart_ratio
}

for dev in ["mobile", "tablet"]:
    input_dict[f"device_type_{dev}"] = 1 if dominant_device == dev else 0

for reg in ["North", "South", "West"]:
    input_dict[f"region_{reg}"] = 1 if region == reg else 0

input_df = pd.DataFrame([input_dict])
input_scaled = scaler.transform(input_df)

if st.button("🚦 Predict Conversion"):
    prediction = model.predict(input_scaled)[0]
    st.write(f"**Prediction:** {'Will Purchase' if prediction == 1 else 'Will not Purchase'}")
