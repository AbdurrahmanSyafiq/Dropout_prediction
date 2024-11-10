import streamlit as st
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load model .h5
model = tf.keras.models.load_model('model/model.h5')

# Title
st.title("Student Status Prediction")

# Function to create inputs based on min and max values
def create_input(label, min_val, max_val, input_type="number"):
    return st.number_input(label, min_value=min_val, max_value=max_val) if input_type == "number" else st.selectbox(label, [0, 1])

# Input fields
Application_order = create_input("Application_order", 0, 6)
Previous_qualification_grade = create_input("Previous_qualification_grade", 95.0, 190.0)
Admission_grade = create_input("Admission_grade", 95.0, 190.0)
Displaced_0 = create_input("Displaced_0", 0, 1, "select")
Displaced_1 = create_input("Displaced_1", 0, 1, "select")
Educational_special_needs_0 = create_input("Educational_special_needs_0", 0, 1, "select")
Educational_special_needs_1 = create_input("Educational_special_needs_1", 0, 1, "select")
Debtor_0 = create_input("Debtor_0", 0, 1, "select")
Debtor_1 = create_input("Debtor_1", 0, 1, "select")
Tuition_fees_up_to_date_0 = create_input("Tuition_fees_up_to_date_0", 0, 1, "select")
Tuition_fees_up_to_date_1 = create_input("Tuition_fees_up_to_date_1", 0, 1, "select")
Scholarship_holder_0 = create_input("Scholarship_holder_0", 0, 1, "select")
Scholarship_holder_1 = create_input("Scholarship_holder_1", 0, 1, "select")
Curricular_units_1st_sem_enrolled = create_input("Curricular_units_1st_sem_enrolled", 0, 26)
Curricular_units_1st_sem_evaluations = create_input("Curricular_units_1st_sem_evaluations", 0, 45)
Curricular_units_1st_sem_approved = create_input("Curricular_units_1st_sem_approved", 0, 26)
Curricular_units_1st_sem_grade = create_input("Curricular_units_1st_sem_grade", 0.0, 18.87)
Curricular_units_1st_sem_without_evaluations = create_input("Curricular_units_1st_sem_without_evaluations", 0, 12)
Curricular_units_2nd_sem_enrolled = create_input("Curricular_units_2nd_sem_enrolled", 0, 23)
Curricular_units_2nd_sem_evaluations = create_input("Curricular_units_2nd_sem_evaluations", 0, 33)
Curricular_units_2nd_sem_approved = create_input("Curricular_units_2nd_sem_approved", 0, 20)
Curricular_units_2nd_sem_grade = create_input("Curricular_units_2nd_sem_grade", 0.0, 18.57)
Curricular_units_2nd_sem_without_evaluations = create_input("Curricular_units_2nd_sem_without_evaluations", 0, 12)
Unemployment_rate = create_input("Unemployment_rate", 7.6, 16.2)
Inflation_rate = create_input("Inflation_rate", -0.8, 3.7)
GDP = create_input("GDP", -4.06, 3.51)

# Create a DataFrame for prediction input
data = {
    "Application_order": [Application_order],
    "Previous_qualification_grade": [Previous_qualification_grade],
    "Admission_grade": [Admission_grade],
    "Displaced_0": [Displaced_0],
    "Displaced_1": [Displaced_1],
    "Educational_special_needs_0": [Educational_special_needs_0],
    "Educational_special_needs_1": [Educational_special_needs_1],
    "Debtor_0": [Debtor_0],
    "Debtor_1": [Debtor_1],
    "Tuition_fees_up_to_date_0": [Tuition_fees_up_to_date_0],
    "Tuition_fees_up_to_date_1": [Tuition_fees_up_to_date_1],
    "Scholarship_holder_0": [Scholarship_holder_0],
    "Scholarship_holder_1": [Scholarship_holder_1],
    "Curricular_units_1st_sem_enrolled": [Curricular_units_1st_sem_enrolled],
    "Curricular_units_1st_sem_evaluations": [Curricular_units_1st_sem_evaluations],
    "Curricular_units_1st_sem_approved": [Curricular_units_1st_sem_approved],
    "Curricular_units_1st_sem_grade": [Curricular_units_1st_sem_grade],
    "Curricular_units_1st_sem_without_evaluations": [Curricular_units_1st_sem_without_evaluations],
    "Curricular_units_2nd_sem_enrolled": [Curricular_units_2nd_sem_enrolled],
    "Curricular_units_2nd_sem_evaluations": [Curricular_units_2nd_sem_evaluations],
    "Curricular_units_2nd_sem_approved": [Curricular_units_2nd_sem_approved],
    "Curricular_units_2nd_sem_grade": [Curricular_units_2nd_sem_grade],
    "Curricular_units_2nd_sem_without_evaluations": [Curricular_units_2nd_sem_without_evaluations],
    "Unemployment_rate": [Unemployment_rate],
    "Inflation_rate": [Inflation_rate],
    "GDP": [GDP]
}

input_df = pd.DataFrame(data)

# Prediction button
if st.button("Predict"):
    prediction = model.predict(input_df)
    st.write("Predicted Status:", "Dropout" if prediction[0][0] > 0.5 else "Not Dropout")
