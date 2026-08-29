import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Indian Bank Loan Eligibility Portal", page_icon="🏦", layout="wide")

@st.cache_resource
def load_artifacts():
    model_path = os.path.join("src", "best_model.pkl")
    scaler_path = os.path.join("src", "scaler.pkl")
    encoders_path = os.path.join("src", "encoders.pkl")

    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(encoders_path)):
        return None, None, None

    return joblib.load(model_path), joblib.load(scaler_path), joblib.load(encoders_path)

model, scaler, encoders = load_artifacts()

st.image("https://upload.wikimedia.org/wikipedia/commons/e/e7/Indian_Bank_logo.svg", width=300)
st.title("🏦 Indian Bank Loan Eligibility Portal")
st.write("Enter financial and demographic details to evaluate real-time loan approval eligibility.")

if model is None:
    st.error("Model artifacts not found! Run preprocessing.py and train_model.py first.")
else:
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

        # Financial Hard-Rule Guardrails (Overwrites ML model on obvious baseline risks)
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
