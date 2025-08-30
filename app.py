import streamlit as st
import joblib
import numpy as np

# Load your trained model
model = joblib.load("model.pkl")

# App Title
st.title("⚡ XGBoost Energy Prediction App")
st.write("Predict energy consumption using Temperature & Humidity sensor data")

# Input fields
temp = st.number_input("🌡️ Enter Temperature (°C)", value=25.0)
hum = st.number_input("💧 Enter Humidity (%)", value=50.0)

# Prediction button
if st.button("🔮 Predict"):
    features = np.array([[temp, hum]])
    prediction = model.predict(features)
    st.success(f"🔋 Predicted Energy Consumption: {prediction[0]:.2f}")
