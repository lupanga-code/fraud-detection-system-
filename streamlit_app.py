import streamlit as st
import pandas as pd
import joblib
import datetime

# Load simplified model
model = joblib.load("fraud_detection_model_simple.pkl")

st.title("💳 Fraud Detection System (Minimal Features)")
st.write("Enter transaction details below to check if it's fraudulent.")

# Input fields
amount = st.number_input("Transaction Amount", min_value=1.0, step=1.0)
merchant = st.text_input("Merchant (e.g., ShopA, ShopB)")
location = st.text_input("Location (e.g., Dar-es-Salaam, Dodoma)")
channel = st.selectbox("Channel", ["POS", "Web", "ATM", "MobileApp"])
timestamp = st.time_input("Transaction Time (HH:MM)", datetime.time(12, 0))

# Feature engineering (Hour)
hour = timestamp.hour

# Encode categorical values
merchant_map = {"ShopA": 0, "ShopB": 1, "ShopC": 2, "ShopD": 3}
location_map = {"Dar-es-Salaam": 0, "Dodoma": 1, "Arusha": 2}
channel_map = {"POS": 0, "Web": 1, "ATM": 2, "MobileApp": 3}

merchant_encoded = merchant_map.get(merchant, 0)
location_encoded = location_map.get(location, 0)
channel_encoded = channel_map.get(channel, 0)

# Build input data with minimal features
features_dict = {
    "Amount": amount,
    "Merchant": merchant_encoded,
    "Location": location_encoded,
    "Channel": channel_encoded,
    "Hour": hour
}

input_data = pd.DataFrame([[features_dict[f] for f in model.feature_names_in_]],
                          columns=model.feature_names_in_)

# Predict button
if st.button("Predict Fraud"):
    proba = model.predict_proba(input_data)[0][1]  # probability of fraud
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected! (Probability: {proba:.2f})")
    else:
        st.success(f"✅ Transaction is Legitimate. (Probability: {proba:.2f})")
