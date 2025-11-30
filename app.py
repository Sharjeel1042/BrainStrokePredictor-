import joblib
import streamlit as st
import pandas as pd
from prediction import predictFromuserInput

st.title("Brain Stroke Predictor")



gender=st.radio("Patient's Gender",["Male","Female"])

age=st.slider("Patient's Age",0,100,50)

hypertension=st.radio("Does the patient have hypertension?",("Yes","No"))

heartDisease=st.radio("Is the patient suffering from any heart disease?",("Yes","No"))

everMarried=st.radio("Is/was the patient married?",("Yes","No"))

workType=st.radio("What is the patient's job?",("children","Private","Self-employed","Govt_Job","Never_worked"))

residenceType=st.radio("What is the patient's residence type?",("Urban","Rural"))

avgGlucoseLevel=st.number_input("Average Glucose Level(mg/dL)",0.0,400.0,100.0,0.1)

bmi=st.number_input("BMI (Body Mass Index)",5.0,110.0,20.0,0.1)

smokingStatus=st.radio("Patient's Smoking Status",("never smoked","smokes","formerly smoked"))

userInput={
    "gender":gender,
    "age":age,
    "hypertension":1 if hypertension=="Yes" else 0,
    "heart_disease":1 if heartDisease=="Yes" else 0,
    "ever_married":everMarried,
    "work_type":workType,
    "Residence_type":residenceType,
    "avg_glucose_level":avgGlucoseLevel,
    "bmi":bmi,
    "smoking_status":smokingStatus
}


if st.button("Predict"):
    pred,prob=predictFromuserInput(userInput)

    stroke="Yes" if pred[0]==1 else "No"
    st.success(f"Patient has stroke: {stroke}")
    st.success(f"Confidence of patient not having stroke: {prob[0,0]*100:.2f}%")
    st.success(f"Confidence of patint having stroke: {prob[0,1]*100:.2f}%")
