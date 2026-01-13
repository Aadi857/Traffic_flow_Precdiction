import os
import warnings

# 1. Suppress TensorFlow Info and Warning logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# 2. Suppress Python/NumPy Deprecation Warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import tensorflow as tf
import numpy as np
# ... rest of your imports (streamlit, etc.)

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# ==============================
# Load Model and Scaler
# ==============================
@st.cache_resource
def load_artifacts():
    model = load_model('lstm_traffic_model.h5')
    scaler = joblib.load('my_scaler.save')
    train_df = pd.read_csv('train_df.csv')  # to get feature order
    return model, scaler, train_df

model, scaler, train_df = load_artifacts()
train_features = train_df.columns.tolist()

# ==============================
# Streamlit Page Setup
# ==============================
st.set_page_config(page_title="Real-Time Traffic Flow Prediction", page_icon="🚗", layout="wide")

st.title("🚦 Real-Time Traffic Flow Prediction using LSTM Network")
st.markdown("Enter environmental and temporal data below to predict **traffic volume**.")

# ==============================
# User Input Section
# ==============================
col1, col2, col3 = st.columns(3)

with col1:
    hour = st.slider("Hour of Day (0–23)", 0, 23, 8)
    weekday = st.slider("Day of Week (1–7)", 1, 7, 3)
    month = st.slider("Month (1–12)", 1, 12, 5)

with col2:
    temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0)
    windspeed = st.number_input("Wind Speed (km/h)", 0.0, 100.0, 15.0)

with col3:
    weather_condition = st.selectbox("Weather Condition", ["Clear", "Clouds", "Rain"])

# ==============================
# Build Input Feature Dictionary
# ==============================
test_data_dict = {
    "hour": [hour],
    "weekday": [weekday],
    "month": [month],
    "temp": [temperature],
    "humidity": [humidity],
    "wind_speed": [windspeed],
    "weather_Clear": [1 if weather_condition == "Clear" else 0],
    "weather_Clouds": [1 if weather_condition == "Clouds" else 0],
    "weather_Rain": [1 if weather_condition == "Rain" else 0],
}
#Day 'sin/cos' columns can be added similarly if they were part of training features
import math
test_data_dict["Day sin"] = [math.sin(2 * math.pi * (weekday / 7))]
test_data_dict["Day cos"] = [math.cos(2 * math.pi * (weekday / 7))]

# Fill other features (if any) with zeros or dummy values
for f in train_features:
    if f not in test_data_dict:
        test_data_dict[f] = [0]

# Create DataFrame in correct feature order
input_df = pd.DataFrame(test_data_dict)[train_features]

# ==============================
# Prediction
# ==============================
if st.button("🔍 Predict Traffic Volume"):
    scaled_input = scaler.transform(input_df)
    scaled_input = np.expand_dims(scaled_input, axis=1)  # [batch, time, features]
    scaled_prediction = model.predict(scaled_input)

    # Denormalize prediction
    min_traffic_volume, max_traffic_volume = 0, 7280
    predicted_volume = scaled_prediction * (max_traffic_volume - min_traffic_volume) + min_traffic_volume
    predicted_volume = float(np.maximum(predicted_volume, 0))

    st.success(f"🚗 **Predicted Traffic Volume:** {predicted_volume:.2f} vehicles/hour")

    st.write("### Input Summary")
   #st.dataframe(input_df)
    st.table(input_df)


#  run the file 
# streamlit run demo.py
#python -m streamlit run demo.py
#PS C:\Users\talva> cd "C:\Users\talva\OneDrive\Desktop\finalYearProject\finalProject"
#PS C:\Users\talva\OneDrive\Desktop\finalYearProject\finalProject> python -m streamlit run demo.py