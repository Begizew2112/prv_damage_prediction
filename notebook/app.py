import streamlit as st
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model

# Load Keras model
model_path = os.path.join(os.path.dirname(__file__), 'rul_model1.keras')
model = load_model(model_path)

st.set_page_config(page_title="Valve Damage Prediction", layout="centered")
st.title("🔧 Pressure Regulating Valve Damage Predictor")
st.write("Upload CSV data containing x and y direction vibration signals.")

# File uploader
uploaded_file = st.file_uploader("Upload Vibration CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)

        # Preview
        st.subheader("📊 Uploaded Data Preview")
        st.write(df.head())

        # Check required columns
        if not {'vibration_x', 'vibration_y'}.issubset(df.columns):
            st.error("CSV must contain 'vibration_x' and 'vibration_y' columns.")
        else:
            # Extract features
            input_data = df[['vibration_x', 'vibration_y']].to_numpy()

            # Reshape if needed (depends on your model input)
            # Example: (n_samples, 2) → model might expect (1, n_samples, 2)
            input_data = np.expand_dims(input_data, axis=0)  # shape = (1, n_samples, 2)

            # Predict
            prediction = model.predict(input_data)
            predicted_value = float(prediction[0][0])

            # Display results
            st.subheader("🧠 Prediction Result")
            if predicted_value < 0.3:
                st.success(f"Low Risk of Failure: {predicted_value:.2f}", icon="✅")
            elif predicted_value < 0.7:
                st.warning(f"Moderate Risk of Failure: {predicted_value:.2f}", icon="⚠️")
            else:
                st.error(f"High Risk of Failure: {predicted_value:.2f}", icon="❌")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
else:
    st.info("Please upload a CSV file to begin.")
