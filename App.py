import streamlit as st
import numpy as np
import joblib

st.title("Fraud Detection Model")


model = joblib.load('fraud_detection_model.joblib')

st.subheader("Enter Transaction Details")

transaction_type = st.selectbox("Transaction Type", ["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"])
amount = st.number_input("Transaction Amount", min_value=0.0)
old_balance_org = st.number_input("Original Balance Before Transaction", min_value=0.0)
new_balance_orig = st.number_input("New Balance After Transaction", min_value=0.0)

transaction_type_mapping = {
    "CASH_OUT": 1,
    "PAYMENT": 2,
    "CASH_IN": 3,
    "TRANSFER": 4,
    "DEBIT": 5
}

if st.button("Predict Fraud"):
    input_data = np.array([[transaction_type_mapping[transaction_type], amount, old_balance_org, new_balance_orig]])
    st.write("Input data for prediction:", input_data)

    prediction = model.predict(input_data)

    st.write("prediction output:", prediction)
