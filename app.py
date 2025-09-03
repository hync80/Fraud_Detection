import streamlit as st
import pandas as pd
import pickle

# Load trained model
model = pickle.load(open('rf_model.pkl', 'rb'))

st.title("Customer Churn Predictor")

# --- Input fields for all model features (update these as per your X_train.columns) ---
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 1, 72)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["No internet service", "No", "Yes"])
online_backup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])
device_protection = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])
tech_support = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])
streaming_tv = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
streaming_movies = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox(
    "Payment Method", 
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, step=1.0)
total_charges = st.number_input("Total Charges", min_value=0.0, step=1.0)

# --- Encoding (must match your training pipeline) ---
gender_encoded = 1 if gender == "Male" else 0
senior_citizen_encoded = 1 if senior_citizen == "Yes" else 0
partner_encoded = 1 if partner == "Yes" else 0
dependents_encoded = 1 if dependents == "Yes" else 0
phone_service_encoded = 1 if phone_service == "Yes" else 0
multiple_lines_encoded = {"No phone service": 0, "No": 1, "Yes": 2}[multiple_lines]
internet_service_encoded = {"DSL": 0, "Fiber optic": 1, "No": 2}[internet_service]
online_security_encoded = {"No internet service": 0, "No": 1, "Yes": 2}[online_security]
online_backup_encoded = {"No internet service": 0, "No": 1, "Yes": 2}[online_backup]
device_protection_encoded = {"No internet service": 0, "No": 1, "Yes": 2}[device_protection]
tech_support_encoded = {"No internet service": 0, "No": 1, "Yes": 2}[tech_support]
streaming_tv_encoded = {"No internet service": 0, "No": 1, "Yes": 2}[streaming_tv]
streaming_movies_encoded = {"No internet service": 0, "No": 1, "Yes": 2}[streaming_movies]
contract_encoded = {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract]
paperless_billing_encoded = 1 if paperless_billing == "Yes" else 0
payment_method_encoded = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer (automatic)": 2,
    "Credit card (automatic)": 3
}[payment_method]

# --- Build input DataFrame in same order as training ---
input_df = pd.DataFrame(
    [[gender_encoded, senior_citizen_encoded, partner_encoded, dependents_encoded, tenure,
      phone_service_encoded, multiple_lines_encoded, internet_service_encoded, online_security_encoded,
      online_backup_encoded, device_protection_encoded, tech_support_encoded, streaming_tv_encoded,
      streaming_movies_encoded, contract_encoded, paperless_billing_encoded, payment_method_encoded,
      monthly_charges, total_charges]],
    columns=["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", "MultipleLines",
             "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
             "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
             "MonthlyCharges", "TotalCharges"]
)

st.write("Input data for prediction:", input_df)

if st.button("Predict"):
    try:
        prediction = model.predict(input_df)
        st.write("Prediction:", "Churn" if prediction[0] == 1 else "No Churn")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

