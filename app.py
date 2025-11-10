import streamlit as st
import numpy as np
import joblib

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("profit_predictor.pkl")

model = load_model()

# UI Layout
st.set_page_config(page_title="Profit Prediction App", layout="centered")
st.title("💰 Profit Prediction App")
st.subheader("Enter company spending values to predict profit:")

# Input fields
rd = st.number_input("R&D Spend (₹)", min_value=0.0, step=1000.0)
admin = st.number_input("Administration Spend (₹)", min_value=0.0, step=1000.0)
marketing = st.number_input("Marketing Spend (₹)", min_value=0.0, step=1000.0)

# Predict button
if st.button("Predict Profit"):
    features = np.array([[rd, admin, marketing]])
    profit = model.predict(features)[0]
    st.success(f"📈 Predicted Profit: ₹{profit:,.2f}")
