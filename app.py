import streamlit as st
import pandas as pd
import joblib

model=joblib.load('LogisticRegression_model.pkl')
scaler=joblib.load('scaler.pkl')
expected_columns=joblib.load('columns.pkl')

st.title('Heart Disease Prediction')
st.markdown('Provide the following details to predict the risk of heart disease')

age=st.slider('Age',18,100,40)
sex=st.selectbox('Sex', ['M', 'F'])
chest_pain_type=st.selectbox('Chest Pain Type', ['TA', 'ATA', 'NAP', 'ASY'])
resting_blood_pressure=st.number_input('Resting Blood Pressure(mm Hg)', 80, 200, 120)
cholesterol=st.number_input('Cholesterol(mg/dl)', 100, 600, 200)
fasting_blood_sugar=st.selectbox('Fasting Blood Sugar > 120 mg/dl',[0,1])
resting_ecg=st.selectbox('Resting ECG', ['Normal', 'ST', 'LVH'])
max_heart_rate=st.slider('Max Heart Rate', 60, 220, 150)
exercise_induced_angina=st.selectbox('Exercise Induced Angina', ['Y','N'])
oldpeak=st.slider('Oldpeak (ST Depression)', 0.0, 6.0, 1.0)
st_slope=st.selectbox('Slope of ST Segment', ['Up', 'Flat', 'Down'])


if st.button('Predict'):
    raw_input={
        'Age': age,
        'RestingBP': resting_blood_pressure,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_blood_sugar,
        'MaxHR': max_heart_rate,
        'Oldpeak': oldpeak,
        'Sex_'+sex: 1,
        'ChestPainType_'+chest_pain_type: 1,
        'RestingECG_'+resting_ecg: 1,
        'ExerciseInducedAngina_'+exercise_induced_angina: 1,
        'ST_Slope_'+st_slope: 1
    }

    input_df=pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col]=0

    input_df=input_df[expected_columns]

    scaled_input=scaler.transform(input_df)
    prediction=model.predict(scaled_input)[0]

    if prediction== 1:
        st.error(' High Risk!!!!!!!')
    else:
      st.success('Low Risk- You are safe :))')