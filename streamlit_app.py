import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("fraud_detection_model_smote.pkl")


st.title("💳 Fraud Detection System")
st.write("Enter transaction details below to check if it's fraudulent.")

# Input fields
amount = st.number_input("Transaction Amount", min_value=1.0, step=1.0)
merchant = st.text_input("Merchant (e.g., ShopA, ShopB)")
location = st.text_input("Location (e.g., Dar-es-Salaam, Dodoma)")
channel = st.selectbox("Channel", ["POS", "Web", "ATM", "MobileApp"])

# Encode categorical values (simple mapping for demo)
def encode_value(value, mapping):
    return mapping.get(value, 0)

merchant_map = {"ShopA": 0, "ShopB": 1, "ShopC": 2, "ShopD": 3}
location_map = {"Dar-es-Salaam": 0, "Dodoma": 1, "Arusha": 2}
channel_map = {"POS": 0, "Web": 1, "ATM": 2, "MobileApp": 3}

merchant_encoded = encode_value(merchant, merchant_map)
location_encoded = encode_value(location, location_map)
channel_encoded = encode_value(channel, channel_map)

# Predict button
if st.button("Predict Fraud"):
    input_data = pd.DataFrame([[amount, merchant_encoded, location_encoded, channel_encoded]],
                              columns=["Amount", "Merchant", "Location", "Channel"])
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Transaction is Legitimate.")
