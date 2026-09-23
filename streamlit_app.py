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

# Build dictionary of all features
features_dict = {
    "Amount": amount,
    "Merchant": merchant_encoded,
    "Location": location_encoded,
    "Channel": channel_encoded,
    "Hour": hour,
    "UnusualHour": unusual_hour,
    "TxnCountPerMinute": txn_count_per_minute,
    "MultiAccountFlag": int(multi_account_flag),
    "LocationMismatch": int(location_mismatch),
    "ChannelSwitch": int(channel_switch)
}

# Align with model.feature_names_in_
input_data = pd.DataFrame([[features_dict.get(f, 0) for f in model.feature_names_in_]],
                          columns=model.feature_names_in_)

# Predict button
if st.button("Predict Fraud"):
    proba = model.predict_proba(input_data)[0][1]  # probability of fraud
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected! (Probability: {proba:.2f})")
    else:
        st.success(f"✅ Transaction is Legitimate. (Probability: {proba:.2f})")
