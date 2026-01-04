import streamlit as st
import pandas as pd
import joblib 

model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

st.title("Heart Disease Prediction App❤️")
st.markdown("This app predicts the presence of heart disease based on user inputs.")

age = st.slider("Age", 18,100,40)
sex = st.selectbox("SEX",['M','F'])
cp = st.selectbox("Chest Pain Type",['TA','ATA','NAP','ASY'])
trestbps = st.slider("Resting Blood Pressure (in mm Hg)", 80,200,120)
chol = st.slider("Serum Cholesterol (in mg/dl)", 100,600,200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl",[0,1])
restecg = st.selectbox("Resting Electrocardiographic Results",['Normal','ST','LVH'])
thalach = st.slider("Maximum Heart Rate Achieved", 60,220,150)
exang = st.selectbox("Exercise Induced Angina",['Y','N'])
oldpeak = st.slider("ST Depression Induced by Exercise", 0.0,6.0,1.0)
slope = st.selectbox("Slope of the Peak Exercise ST Segment",['Up','Flat','Down'])


if st.button("Predict"):
    raw_input = {
        'Age': age,
        'Sex_' + sex: 1,
        'CP_' + cp: 1,
        'Trestbps': trestbps,
        'Chol': chol,
        'Fbs': fbs,
        'Restecg_' + restecg: 1,
        'Thalach': thalach,
        'Exang_' + exang: 1,
        'Oldpeak': oldpeak,
        'Slope_' + slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    if prediction == 1:
        st.error("⚠️The model predicts that you may have heart disease. Please consult a healthcare professional for further evaluation.")
    else:
        st.success("✅The model predicts that you are unlikely to have heart disease. Keep maintaining a healthy lifestyle!")