import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Page settings
st.set_page_config(page_title="Forest Fire Dashboard", layout="centered")

# Title
st.title("🌲 Forest Fire Risk Prediction Dashboard")
st.markdown("Upload data or manually enter values to predict fire risk using a trained ML model.")

# Load the trained model
model = joblib.load("model.pkl")

# --- File Upload Section ---
uploaded_file = st.file_uploader("📁 Upload your CSV data", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data", data.head())

    if 'risk_score' in data.columns:
        st.subheader("📊 Fire Risk Score Chart")
        st.bar_chart(data['risk_score'])

# --- Manual Input Section ---
st.subheader("🧠 Manual Fire Risk Prediction")

col1, col2, col3 = st.columns(3)
with col1:
    temp = st.number_input("🌡️ Temperature (°C)", min_value=0)
with col2:
    humidity = st.number_input("💧 Humidity (%)", min_value=0)
with col3:
    wind = st.number_input("🍃 Wind Speed (km/h)", min_value=0)

if st.button("Predict Fire Risk"):
    input_data = [[temp, humidity, wind]]
    prediction = model.predict(input_data)
    st.success(f"🔥 Predicted Fire Risk: {prediction[0]}")





