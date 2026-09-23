import streamlit as st
import pandas as pd
import joblib
import datetime

# Load trained model
model = joblib.load("fraud_detection_model.pkl")

st.title("💳 Fraud Detection System")
st.write("Enter transaction details below to check if it's fraudulent.")

# Input fields
amount = st.number_input("Transaction Amount", min_value=1.0, step=1.0)
merchant = st.text_input("Merchant (e.g., ShopA, ShopB)")
location = st.text_input("Location (e.g., Dar-es-Salaam, Dodoma)")
channel = st.selectbox("Channel", ["POS", "Web", "ATM", "MobileApp"])
timestamp = st.time_input("Transaction Time (HH:MM)", datetime.time(12, 0))
multi_account_flag = st.checkbox("Multiple accounts used?")
location_mismatch = st.checkbox("Location mismatch?")
channel_switch = st.checkbox("Channel switch?")

# Feature engineering
hour = timestamp.hour
unusual_hour = 1 if hour < 6 or hour > 22 else 0
txn_count_per_minute = amount / 60.0  # simple proxy example

# Encode categorical values
merchant_map = {"ShopA": 0, "ShopB": 1, "ShopC": 2, "ShopD": 3}
location_map = {"Dar-es-Salaam": 0, "Dodoma": 1, "Arusha": 2}
channel_map = {"POS": 0, "Web": 1, "ATM": 2, "MobileApp": 3}

merchant_encoded = merchant_map.get(merchant, 0)
location_encoded = location_map.get(location, 0)
channel_encoded = channel_map.get(channel, 0)

# Build full feature set
input_data = pd.DataFrame([[
    amount,
    merchant_encoded,
    location_encoded,
    channel_encoded,
    hour,
    unusual_hour,
    txn_count_per_minute,
    int(multi_account_flag),
    int(location_mismatch),
    int(channel_switch)
]], columns=[
    "Amount",
    "Merchant",
    "Location",
    "Channel",
    "Hour",
    "UnusualHour",
    "TxnCountPerMinute",
    "MultiAccountFlag",
    "LocationMismatch",
    "ChannelSwitch"
])

# Predict button
if st.button("Predict Fraud"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Transaction is Legitimate.")
