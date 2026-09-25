import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# 1. Load model na dataset (cached)
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("fraud_detection_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("synthetic_dataset.csv")

model = load_model()
df = load_data()

# -------------------------------
# 2. App layout
# -------------------------------
st.title("Fraud Detection App 🚨")
st.write("Hii app inatumia Logistic Regression/Random Forest kutabiri kama transaction ni fraud au sio.")

# Show dataset preview
st.subheader("Dataset Preview")
st.dataframe(df.head())

# -------------------------------
# 3. User input form
# -------------------------------
st.subheader("Make a Prediction")

amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0)
merchant = st.selectbox("Merchant", df["Merchant"].unique())
location = st.selectbox("Location", df["Location"].unique())
channel = st.selectbox("Channel", df["Channel"].unique())

# Encode categorical inputs (simple mapping)
input_data = pd.DataFrame({
    "Amount": [amount],
    "Merchant": [merchant],
    "Location": [location],
    "Channel": [channel]
})

# Convert categorical to numeric (same encoding used in training)
input_data = pd.get_dummies(input_data)
# Align with training columns
input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

# -------------------------------
# 4. Prediction
# -------------------------------
if st.button("Predict Fraud"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Fraud detected! Probability: {probability:.2f}")
    else:
        st.success(f"✅ Legit transaction. Probability of fraud: {probability:.2f}")
