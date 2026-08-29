import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Get absolute path of script's directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def create_mock_data():
    np.random.seed(42)
    n = 600
    data = {
        'Gender': np.random.choice(['Male', 'Female'], n),
        'Married': np.random.choice(['Yes', 'No'], n),
        'Dependents': np.random.choice(['0', '1', '2', '3+'], n),
        'Education': np.random.choice(['Graduate', 'Not Graduate'], n),
        'Self_Employed': np.random.choice(['No', 'Yes'], n),
        'ApplicantIncome': np.random.randint(15000, 150000, n),
        'CoapplicantIncome': np.random.randint(0, 50000, n),
        'LoanAmount': np.random.randint(100, 500, n),
        'Loan_Amount_Term': np.random.choice([360, 180, 240, 300, 480], n),
        'Credit_History': np.random.choice([1.0, 0.0], n, p=[0.8, 0.2]),
        'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], n)
    }
    df = pd.DataFrame(data)
    
    total_income = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['Loan_Status'] = np.where(
        (df['Credit_History'] == 1.0) & (df['LoanAmount'] * 1000 <= total_income * 36), 1, 0
    )
    return df

def preprocess():
    df = create_mock_data()
    
    encoders = {}
    cat_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        
    df['Dependents'] = df['Dependents'].str.replace('+', '', regex=False).astype(int)

    scaler = StandardScaler()
    scale_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    processed_dir = os.path.join(BASE_DIR, "data", "processed")
    src_dir = os.path.join(BASE_DIR, "src")
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(src_dir, exist_ok=True)

    df.to_csv(os.path.join(processed_dir, "train_processed.csv"), index=False)
    joblib.dump(scaler, os.path.join(src_dir, "scaler.pkl"))
    joblib.dump(encoders, os.path.join(src_dir, "encoders.pkl"))
    print("✅ Preprocessing complete. Files saved successfully.")

if __name__ == "__main__":
    preprocess()
