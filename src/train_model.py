import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train():
    processed_path = os.path.join("data", "processed", "train_processed.csv")
    model_path = os.path.join("src", "best_model.pkl")

    df = pd.read_csv(processed_path)
    X = df.drop(columns=["Loan_Status"])
    y = df["Loan_Status"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"✅ Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

    joblib.dump(model, model_path)
    print("✅ Model trained and saved.")

if __name__ == "__main__":
    train()
