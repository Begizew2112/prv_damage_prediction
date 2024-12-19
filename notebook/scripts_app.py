import streamlit as st
import joblib
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# Load the model using relative path
model_path = os.path.join(os.path.dirname(__file__), 'pressure_regulating_valve_model.joblib')
model = joblib.load(model_path)

# Your Streamlit app code follows...

# Define the Streamlit dashboard
st.title("Pressure Regulating Valve Damage Prediction")
st.sidebar.header("Input Parameters")

# Numerical input parameters
Pressure_Input = st.sidebar.number_input(
    'Pressure Input (kPa)', min_value=0, max_value=5000, value=1000, step=10
)
Pressure_Output = st.sidebar.number_input(
    'Pressure Output (kPa)', min_value=0, max_value=5000, value=450, step=10
)
Pressure_Difference = st.sidebar.number_input(
    'Pressure Difference (kPa)', min_value=0, max_value=500, value=550, step=1
)
Flow_Rate = st.sidebar.number_input(
    'Flow Rate (m³/s)', min_value=0.0, max_value=10.0, value=6.0, step=0.1
)
Temperature = st.sidebar.number_input(
    'Temperature (°C)', min_value=-50, max_value=90, value=50, step=1
)
Operating_Time = st.sidebar.number_input(
    'Operating Time (Hours)', min_value=0, max_value=50000, value=3000, step=100
)
Valve_Size = st.sidebar.number_input(
    'Valve Size (mm)', min_value=10, max_value=200, value=100, step=5
)
Maintenance_Frequency = st.sidebar.number_input(
    'Maintenance Frequency (Months)', min_value=0, max_value=40, value=12, step=1
)
Load_Cycles = st.sidebar.number_input(
    'Load Cycles', min_value=0, max_value=999079, value=20000, step=100
)

# Dropdown menus for categorical inputs
Material_Type = st.sidebar.selectbox(
    'Material Type', ['Brass', 'Polymer', 'Steel']
)
Environmental_Conditions = st.sidebar.selectbox(
    'Environmental Conditions', ['Corrosive', 'Humid', 'Normal']
)

# Derived features
Pressure_Ratio = Pressure_Input / max(Pressure_Output, 1)  # Avoid division by zero
Normalized_Pressure_Difference = Pressure_Difference / max(Pressure_Input, 1)

# One-hot encoding logic for categorical inputs
Material_Type_Brass = 1 if Material_Type == 'Brass' else 0
Material_Type_Polymer = 1 if Material_Type == 'Polymer' else 0
Material_Type_Steel = 1 if Material_Type == 'Steel' else 0
Env_Cond_Corrosive = 1 if Environmental_Conditions == 'Corrosive' else 0
Env_Cond_Humid = 1 if Environmental_Conditions == 'Humid' else 0
Env_Cond_Normal = 1 if Environmental_Conditions == 'Normal' else 0

# Input features for prediction
input_features = [
    Pressure_Input, Pressure_Output, Pressure_Difference, Flow_Rate,
    Temperature, Operating_Time, Valve_Size, Maintenance_Frequency,
    Load_Cycles, Material_Type_Brass, Material_Type_Polymer, Material_Type_Steel,
    Env_Cond_Corrosive, Env_Cond_Humid, Env_Cond_Normal, Pressure_Ratio,
    Normalized_Pressure_Difference
]

# Adding custom CSS for button styling
st.markdown(
    """
    <style>
    .stButton button {
        background-color: #4CAF50;  /* Green background */
        color: white;  /* White text */
        font-size: 18px;  /* Font size */
        padding: 12px 24px;  /* Button padding */
        border: none;  /* Remove border */
        border-radius: 8px;  /* Rounded corners */
        cursor: pointer;  /* Cursor pointer on hover */
        transition: background-color 0.3s ease;  /* Smooth transition */
    }

    .stButton button:hover {
        background-color: #45a049;  /* Darker green on hover */
    }
    </style>
    """, unsafe_allow_html=True
)

# Display the "Predict" button with custom style
predict_button = st.button("Predict Damage")

# Predict and display results when the button is clicked
if predict_button:
    try:
        prediction = model.predict([input_features])
        if hasattr(model, "predict_proba"):
            prediction_proba = model.predict_proba([input_features])[0][1]  # Probability of failure
        else:
            prediction_proba = None

        # Display results
        if prediction[0] == 0 and (prediction_proba is None or prediction_proba < 0.2):
            st.success("Result: No Failure Detected (Safe Zone)", icon="✅")
            if prediction_proba is not None:
                st.markdown(
                    f"<h3 style='color: green;'>Probability of Failure: {prediction_proba:.2f}</h3>",
                    unsafe_allow_html=True,
                )
        elif prediction_proba is not None and 0.2 <= prediction_proba <= 0.5:
            st.warning("Result: At Risk (Caution)", icon="⚠️")
            st.markdown(
                f"<h3 style='color: yellow;'>Probability of Failure: {prediction_proba:.2f}</h3>",
                unsafe_allow_html=True,
            )
        else:
            st.error("Result: Failure Likely (Danger)", icon="❌")
            if prediction_proba is not None:
                st.markdown(
                    f"<h3 style='color: red;'>Probability of Failure: {prediction_proba:.2f}</h3>",
                    unsafe_allow_html=True,
                )

        # Feature importance plot
        st.subheader('Feature Importance')
        
        # Get feature importance
        feature_importances = model.feature_importances_  # Replace with your model if necessary
        features = [
            'Pressure_Input', 'Pressure_Output', 'Pressure_Difference', 'Flow_Rate', 'Temperature',
            'Operating_Time', 'Valve_Size', 'Maintenance_Frequency', 'Load_Cycles', 'Material_Type_Brass',
            'Material_Type_Polymer', 'Material_Type_Steel', 'Env_Cond_Corrosive', 'Env_Cond_Humid',
            'Env_Cond_Normal', 'Pressure_Ratio', 'Normalized_Pressure_Difference'
        ]  # Feature names, adjust accordingly
        importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})

        # Sort by importance
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        # Plot the feature importance
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        ax.set_title('Feature Importance')
        ax.invert_yaxis()  # Invert y-axis to have the most important feature at the top

        # Display the plot in Streamlit
        st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
