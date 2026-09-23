import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

def load_data(path="data/synthetic_dataset.csv"):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    for col in ['Merchant', 'Location', 'Channel']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def train_model(df):
    df = preprocess_data(df)
    X = df[['Amount','Merchant','Location','Channel']]
    y = df['FraudLabel']

    model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    model.fit(X, y)

    joblib.dump(model, "models/fraud_detection_model.pkl")
    print("✅ Model trained and saved successfully!")

def main():
    df = load_data()
    train_model(df)

if __name__ == "__main__":
    main()
