
import streamlit as st
import pandas as pd
import joblib
from scipy.sparse import hstack


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💰",
    layout="centered"
)


# -----------------------------
# Load trained model
# -----------------------------
artifacts = joblib.load("loan_prediction_model.pkl")

model = artifacts["model"]
encoder = artifacts["encoder"]
scaler = artifacts["scaler"]

categorical_cols = artifacts["categorical_cols"]
numerical_cols = artifacts["numerical_cols"]


# -----------------------------
# App title
# -----------------------------
st.title("💰 Loan Approval Predictor")

st.write(
    "Enter the applicant's information below to predict "
    "whether the loan is likely to be approved."
)


# -----------------------------
# Applicant information
# -----------------------------
st.subheader("Applicant Information")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

employment_status = st.selectbox(
    "Employment Status",
    ["Employed", "Self-employed", "Unemployed"]
)

annual_income = st.number_input(
    "Annual Income",
    min_value=0,
    value=50000,
    step=1000
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0,
    value=10000,
    step=1000
)

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=850,
    value=650
)

num_dependents = st.number_input(
    "Number of Dependents",
    min_value=0,
    max_value=20,
    value=0,
    step=1
)

existing_loans_count = st.number_input(
    "Existing Loans Count",
    min_value=0,
    max_value=20,
    value=0,
    step=1
)


# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Loan Approval"):

    # Create dataframe from user input
    input_data = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "marital_status": [marital_status],
        "annual_income": [annual_income],
        "loan_amount": [loan_amount],
        "credit_score": [credit_score],
        "num_dependents": [num_dependents],
        "existing_loans_count": [existing_loans_count],
        "employment_status": [employment_status]
    })

    # Numerical preprocessing
    input_num = scaler.transform(
        input_data[numerical_cols]
    )

    # Categorical preprocessing
    input_cat = encoder.transform(
        input_data[categorical_cols]
    )

    # Combine numerical + categorical features
    input_final = hstack([
        input_num,
        input_cat
    ])

    # Make prediction
    prediction = model.predict(input_final)[0]

    # Prediction probability
    probability = model.predict_proba(input_final)[0][1]

    st.divider()

    if prediction == 1:
        st.success("✅ Loan Likely Approved")
    else:
        st.error("❌ Loan Likely Rejected")

    st.metric(
        "Approval Probability",
        f"{probability * 100:.2f}%"
    )

