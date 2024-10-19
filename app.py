import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the pre-trained models
with open('ridge.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Title for the web app
st.title("Forest Fire Prediction")

# Sidebar to input the variables required for prediction
st.sidebar.header("Input Variables")

def get_user_input():
    # Input fields for each variable in the model (excluding BUI and DC, which were dropped)
    Temperature = st.sidebar.number_input("Temperature (Celsius)", min_value=-20.0, max_value=50.0, step=0.1)
    RH = st.sidebar.slider("Relative Humidity (%)", min_value=0, max_value=100, step=1)
    Ws = st.sidebar.slider("WS (Wind Speed)", min_value=0, max_value=100, step=1)
    Rain = st.sidebar.slider("Rain (mm)", min_value=0, max_value=100, step=1)
    FFMC = st.sidebar.slider("FFMC", min_value=0, max_value=100, step=1)
    DMC = st.sidebar.slider("DMC", min_value=0, max_value=100, step=1)
    ISI = st.sidebar.slider("ISI", min_value=0, max_value=100, step=1)
    Classes = st.sidebar.slider("Classes (0 for low, 1 for high)", min_value=0, max_value=1, step=1)
    region = st.sidebar.slider("Region", min_value=0, max_value=100, step=1)
    
    # Create a DataFrame from the user inputs
    data = {'Temperature': Temperature,
            'RH': RH,
            'Ws': Ws,
            'Rain': Rain,
            'FFMC': FFMC,
            'DMC': DMC,
            'ISI': ISI,
            'Classes': Classes,
            'region': region}
    
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
user_input = get_user_input()

# Show the user input in the main area of the app
st.subheader("User Input:")
st.write(user_input)

# Ensure the columns of user_input match the original scaler feature names
# Drop columns BUI and DC since they were not part of the model training
user_input = user_input[['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'ISI', 'Classes', 'region']]

# Standardize the input using the loaded scaler
scaled_input = scaler.transform(user_input)

# Predict the final output using the loaded model
prediction = model.predict(scaled_input)

# Display the prediction result
st.subheader("Prediction:")
st.write(f"Predicted forest fire risk: {prediction[0]}")
