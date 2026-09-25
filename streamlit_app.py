import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# 1. Load model (cached)
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("fraud_detection_model_simple.pkl")

model = load_model()

# -------------------------------
# 2. App layout
# -------------------------------
st.title("Fraud Detection App 🚨")
st.write("Fast lightweight version — shows fraud probability in percentage.")

# -------------------------------
# 3. User input form
# -------------------------------
st.subheader("Make a Prediction")

amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0)
merchant = st.text_input("Merchant", "KahawaCafe")
location = st.text_input("Location", "Dar es Salaam")
channel = st.selectbox("Channel", ["Mobile", "POS", "Online"])

# Prepare input (minimal features)
input_data = pd.DataFrame({
    "Amount": [amount],
    "Merchant": [merchant],
    "Location": [location],
    "Channel": [channel]
})

# Convert categorical to numeric (dummy variables)
input_data = pd.get_dummies(input_data)
input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

# -------------------------------
# 4. Prediction with percentage
# -------------------------------
if st.button("Predict Fraud"):
    prediction = model.predict(input_data)[0]
    probs = model.predict_proba(input_data)[0]
    legit_pct = probs[0] * 100
    fraud_pct = probs[1] * 100

    if prediction == 1:
        st.error(f"⚠️ Fraud detected! Fraud probability: {fraud_pct:.2f}% | Legit: {legit_pct:.2f}%")
    else:
        st.success(f"✅ Legit transaction. Legit probability: {legit_pct:.2f}% | Fraud: {fraud_pct:.2f}%")
