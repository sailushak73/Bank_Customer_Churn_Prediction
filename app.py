import streamlit as st
import pickle
import numpy as np

# Load model and scaler
with open('churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    sc = pickle.load(f)

st.title("Customer Churn Prediction App")
st.write("Predict whether a customer will exit the bank or not using ML")

# Collect input features
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
age = st.number_input("Age", min_value=18, max_value=100, value=35)
tenure = st.slider("Tenure (Years with bank)", 0, 10, 3)
balance = st.number_input("Balance", value=50000.0)
num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
estimated_salary = st.number_input("Estimated Salary", value=50000.0)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])

# Encoding categorical features manually
geo_map = {'France': [1,0], 'Germany': [0,1], 'Spain': [0,0]}  
gender_val = 1 if gender == "Male" else 0
has_cr_card_val = 1 if has_cr_card == "Yes" else 0
is_active_member_val = 1 if is_active_member == "Yes" else 0
geo_values = geo_map[geography]

# Create input array (based on your model’s feature order)
input_data = np.array([[credit_score, age, tenure, balance, num_of_products,
                        has_cr_card_val, is_active_member_val, estimated_salary,
                        gender_val, geo_values[0], geo_values[1]]])

# Scale input
input_scaled = sc.transform(input_data)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_scaled)
    if prediction[0] == 1:
        st.error("The customer is likely to EXIT the bank.")
    else:
        st.success("The customer is likely to STAY with the bank.")
