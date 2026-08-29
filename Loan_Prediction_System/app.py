import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Indian Bank Loan Eligibility Portal", page_icon="🏦", layout="wide")

# Inline artifact generator
def ensure_artifacts():
    os.makedirs("src", exist_ok=True)
    m_path, s_path, e_path = "src/best_model.pkl", "src/scaler.pkl", "src/encoders.pkl"

    if not (os.path.exists(m_path) and os.path.exists(s_path) and os.path.exists(e_path)):
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

        X = df.drop(columns=["Loan_Status"])
        y = df["Loan_Status"]

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        joblib.dump(model, m_path)
        joblib.dump(scaler, s_path)
        joblib.dump(encoders, e_path)

ensure_artifacts()

@st.cache_resource
def load_artifacts():
    return (
        joblib.load("src/best_model.pkl"),
        joblib.load("src/scaler.pkl"),
        joblib.load("src/encoders.pkl")
    )

model, scaler, encoders = load_artifacts()

st.title("🏦 Indian Bank Loan Eligibility Portal")
st.write("Enter financial and demographic details to evaluate real-time loan approval eligibility.")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Applicant Profile")
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])

with col2:
    st.subheader("Financial Profile")
    applicant_income = st.number_input("Applicant Monthly Income (₹)", min_value=0, value=50000, step=5000)
    coapplicant_income = st.number_input("Coapplicant Monthly Income (₹)", min_value=0, value=0, step=5000)
    loan_amount_actual = st.number_input("Requested Loan Amount (₹)", min_value=0, value=500000, step=50000)
    loan_term = st.selectbox("Loan Term (Days)", [360, 180, 240, 300, 480], index=0)

with col3:
    st.subheader("Credit & Property Details")
    credit_history = st.selectbox("Credit History (1 = Good, 0 = Poor)", [1.0, 0.0])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

dep_val = int(dependents.replace("+", "")) if "+" in dependents else int(dependents)

if st.button("Evaluate Application"):
    total_income = float(applicant_income) + float(coapplicant_income)

    if total_income < 10000 or (loan_amount_actual / max(total_income, 1)) > 36:
        st.markdown("---")
        st.error("❌ **Loan Application Rejected by Indian Bank Policy Rules.**")
        st.subheader("💡 Key Reasons for Rejection:")
        if total_income < 10000:
            st.write(f"🚩 **Insufficient Baseline Income**: Combined monthly income (₹{total_income:,.0f}) is below minimum policy threshold (₹10,000 required).")
        if (loan_amount_actual / max(total_income, 1)) > 36:
            st.write(f"🚩 **Excessive Debt-to-Income Ratio**: Requested loan (₹{loan_amount_actual:,.0f}) is too high relative to total monthly income (₹{total_income:,.0f}).")
    else:
        loan_amount_thousands = float(loan_amount_actual) / 1000.0

        raw_inputs = {
            "Gender": gender,
            "Married": married,
            "Dependents": dep_val,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": float(applicant_income),
            "CoapplicantIncome": float(coapplicant_income),
            "LoanAmount": float(loan_amount_thousands),
            "Loan_Amount_Term": float(loan_term),
            "Credit_History": float(credit_history),
            "Property_Area": property_area
        }

        input_df = pd.DataFrame([raw_inputs])
        cat_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']

        for col in cat_cols:
            if col in input_df.columns and col in encoders:
                le = encoders[col]
                input_df[col] = input_df[col].apply(
                    lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else 0
                )

        scale_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
        input_df[scale_cols] = scaler.transform(input_df[scale_cols])

        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0]

        st.markdown("---")
        if prediction == 1:
            st.success(f"✅ **Loan Approved by Indian Bank!** (Confidence: {max(prob)*100:.1f}%)")
        else:
            st.error(f"❌ **Loan Application Rejected.** (Confidence: {max(prob)*100:.1f}%)")
            st.subheader("💡 Key Reasons for Rejection:")
            if credit_history == 0.0:
                st.write("🚩 **Poor Credit History**: Default record detected on previous credit lines.")
            else:
                st.write("🚩 **Automated Risk Assessment**: Financial parameters exceed standard credit risk tolerance bounds.")
