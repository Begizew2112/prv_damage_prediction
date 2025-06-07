import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from keras.models import load_model

# Streamlit UI
st.title("Remaining Useful Life (RUL) Prediction")
uploaded_file = st.file_uploader("Upload your CSV file with 'x_direction' and 'y_direction' columns", type=["csv"])

# Constants
SEQ_LENGTH = 50
FEATURE_COLUMNS = ["x_direction", "y_direction", "RUL", "elapsed_hours",
                   "rolling_mean_x", "rolling_mean_y", "ewma_x", "ewma_y",
                   "delta_x", "delta_y"]

@st.cache_data
def load_scaler(path="minmax_scaler2.pkl"):
    return joblib.load(path)

@st.cache_resource
def load_lstm_model(path="rul_model1.keras"):
    return load_model(path)

def preprocess_data(df):
    # Feature engineering
    total_rows = len(df)
    total_rows
    # Compute elapsed hours
    df['elapsed_hours'] = (np.arange(total_rows) / total_rows) * 128
    # Compute RUL , the total life fo the bearing is 128 hour
    df['RUL'] = 128 - df['elapsed_hours']
    df["rolling_mean_x"] = df["x_direction"].rolling(window=5).mean()
    df["rolling_mean_y"] = df["y_direction"].rolling(window=5).mean()
    df["ewma_x"] = df["x_direction"].ewm(span=5).mean()
    df["ewma_y"] = df["y_direction"].ewm(span=5).mean()
    df["delta_x"] = df["x_direction"].diff()
    df["delta_y"] = df["y_direction"].diff()
    df = df.dropna().reset_index(drop=True)  # Clean NaNs from rolling/diff
    return df

def create_sequences(data, seq_length):
    data = data.to_numpy()
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length, :-1])  # All columns except RUL (last)
        y.append(data[i+seq_length, -1])     # RUL as target
    return np.array(X), np.array(y)

if uploaded_file is not None:
    try:
        # Load and preprocess
        df_raw = pd.read_csv(uploaded_file)
        if not {"x_direction", "y_direction"}.issubset(df_raw.columns):
            st.error("CSV must contain 'x_direction' and 'y_direction' columns.")
        else:
            df_processed = preprocess_data(df_raw)
            scaler = load_scaler()
            scaled_data = scaler.transform(df_processed[FEATURE_COLUMNS])
            df_scaled = pd.DataFrame(scaled_data, columns=FEATURE_COLUMNS)

            # Sequence creation
            X_seq, y_true = create_sequences(df_scaled, SEQ_LENGTH)

            # Load and predict
            model = load_lstm_model()
            y_pred = model.predict(X_seq).flatten()

          #  Plot results
            st.subheader("Predicted vs Actual RUL")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(y_true[:400], label="Actual RUL")
            ax.plot(y_pred[:400], label="Predicted RUL")
            ax.set_xlabel("Sample Index")
            ax.set_ylabel("Remaining Useful Life")
            ax.set_title("Predicted vs Actual RUL")
            ax.legend()
            st.pyplot(fig)
            # Section Header
            st.subheader("System Visualization / Bearing Image")

            # Display Images
            st.image("data/image1.png", caption="Actual Life vs Predicted", use_column_width=True)
            st.image("data/image2.png", caption="Ball Bearing: Actual vs Predicted", use_column_width=True)
            st.image("data/image3.png", caption="Ball Actual Life vs Predicted", use_column_width=True)
    except Exception as e:
        st.error(f"Something went wrong: {e}")
