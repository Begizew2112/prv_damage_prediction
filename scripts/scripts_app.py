import streamlit as st
import joblib
import numpy as np

# Load the trained model
model_filename = r'C:\Users\Yibabe\Desktop\prv_damage_prediction\notebook\pressure_regulating_valve_model.joblib'
model = joblib.load(model_filename)

# Title of the app
st.title('Pressure Regulating Valve Damage Prediction')

# Description of the app
st.write("""
This app predicts the probability of failure for a pressure regulating valve
based on various input parameters.
""")

# Create input fields for user data (corresponding to your columns)
Pressure_Input = st.number_input('Pressure Input (kPa)', min_value=0, max_value=1000, value=500)
Pressure_Output = st.number_input('Pressure Output (kPa)', min_value=0, max_value=1000, value=450)
Pressure_Difference = st.number_input('Pressure Difference (kPa)', min_value=0, max_value=1000, value=50)
Flow_Rate = st.number_input('Flow Rate (m³/s)', min_value=0.0, max_value=10.0, value=0.1)
Temperature = st.number_input('Temperature (°C)', min_value=-50, max_value=100, value=30)
Operating_Time = st.number_input('Operating Time (Hours)', min_value=0, max_value=10000, value=1000)
Valve_Size = st.number_input('Valve Size (mm)', min_value=0, max_value=500, value=100)
Maintenance_Frequency = st.number_input('Maintenance Frequency (Months)', min_value=0, max_value=24, value=12)
Load_Cycles = st.number_input('Load Cycles', min_value=0, max_value=10000, value=500)

# Categorical features encoded as 0 or 1
Material_Type_Brass = st.selectbox('Material Type (Brass)', [0, 1], index=1)
Material_Type_Polymer = st.selectbox('Material Type (Polymer)', [0, 1], index=0)
Material_Type_Steel = st.selectbox('Material Type (Steel)', [0, 1], index=0)

Environmental_Conditions_Corrosive = st.selectbox('Environmental Conditions (Corrosive)', [0, 1], index=0)
Environmental_Conditions_Humid = st.selectbox('Environmental Conditions (Humid)', [0, 1], index=1)
Environmental_Conditions_Normal = st.selectbox('Environmental Conditions (Normal)', [0, 1], index=0)

# Additional Features
Pressure_Ratio = st.number_input('Pressure Ratio', min_value=0.0, max_value=10.0, value=1.0)
Normalized_Pressure_Difference = st.number_input('Normalized Pressure Difference', min_value=0.0, max_value=10.0, value=1.0)

# Prepare the feature array for prediction
features = np.array([[
    Pressure_Input,
    Pressure_Output,
    Pressure_Difference,
    Flow_Rate,
    Temperature,
    Operating_Time,
    Valve_Size,
    Maintenance_Frequency,
    Load_Cycles,
    Material_Type_Brass,
    Material_Type_Polymer,
    Material_Type_Steel,
    Environmental_Conditions_Corrosive,
    Environmental_Conditions_Humid,
    Environmental_Conditions_Normal,
    Pressure_Ratio,
    Normalized_Pressure_Difference
]])

# Make prediction using the model
prediction = model.predict(features)
prediction_proba = model.predict_proba(features)[0][1]  # Probability of failure

# Show results
if st.button('Predict'):
    st.write(f"Prediction: {'Failure' if prediction[0] == 1 else 'No Failure'}")
    st.write(f"Probability of Failure: {prediction_proba:.2f}")
