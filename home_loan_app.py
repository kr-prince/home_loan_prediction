import pandas as pd
import streamlit as st
from joblib import load


# Initialise model objects
if 'randomforest_model' not in st.session_state:
    st.session_state['randomforest_model'] = load('./models/randomforest_model.model')
if 'scaler_model' not in st.session_state:
    st.session_state['scaler_model'] = load('./models/scaler_model.model')
if 'categorical_continuous_map' not in st.session_state:
    st.session_state['categorical_continuous_map'] = load('./models/categorical_continuous_map.dict')

def predict_home_loan_status():
    columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
               'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
               'Credit_History', 'Property_Area']
    X_data = pd.DataFrame(data=[[gender, marital_status, dependents, education, self_employed,
                                applicant_income, coapplicant_income, loan_amount,
                                loan_amount_term, credit_history, property_area]],
                            columns=columns)
    X_data = X_data.fillna(-1)
    categorical_continuous_map = st.session_state['categorical_continuous_map']
    X_data = X_data.replace(categorical_continuous_map)
    continuous_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                          'Loan_Amount_Term', 'Credit_History']
    scaler_model = st.session_state['scaler_model']
    X_data[continuous_columns] = scaler_model.transform(X_data[continuous_columns])
    
    randomforest_model = st.session_state['randomforest_model']
    prediction = randomforest_model.predict_proba(X_data).flatten()
    prediction_no, prediction_yes = prediction
    st.write(f":red[No - {round(prediction_no, 2)*100}%]")
    st.write(f":green[Yes - {round(prediction_yes, 2)*100}%]")


st.header(":blue[Home Loan Prediction App]", divider='blue')
gender = st.selectbox("Gender", ('Male', 'Female'))
marital_status = st.selectbox("Marital Status", ('No', 'Yes'))
dependents = st.selectbox("#Dependents", ('0', '1', '2', '3+'))
education = st.selectbox("Education", ('Graduate', 'Not Graduate'))
self_employed = st.selectbox("Self-Employment Status", ('No', 'Yes'))
applicant_income = st.number_input("Applicant Income", min_value=0.0)
coapplicant_income = st.number_input("Co-Applicant Income", min_value=0.0)
loan_amount = st.number_input("Loan Amount", min_value=0.0)
loan_amount_term = st.number_input("Loan Amount Term", min_value=0)
credit_history = st.slider("Credit history", min_value=0.0, max_value=1.0, step=0.1)
property_area = st.selectbox("Property area", ('Semiurban', 'Urban', 'Rural'))
submit_button = st.button("Submit", type='primary')

if submit_button:
    on_click=predict_home_loan_status()

if __file__ == '__main__':
    pass
