import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("boston_housing_model.pkl")

# Streamlit App
st.title("🏡 Boston Housing Price Prediction")

st.write("Enter the housing details to predict the price.")

# Input fields
CRIM = st.number_input("Crime Rate", min_value=0.0, format="%.5f")
ZN = st.number_input("Residential Land Zone", min_value=0.0, format="%.2f")
INDUS = st.number_input("Non-retail Business Acres", min_value=0.0, format="%.2f")
CHAS = st.selectbox("Charles River (0 = No, 1 = Yes)", [0, 1])
NOX = st.number_input("Nitrogen Oxide Concentration", min_value=0.0, format="%.3f")
RM = st.number_input("Avg Rooms per Dwelling", min_value=0.0, format="%.2f")
AGE = st.number_input("Age of Property", min_value=0.0, format="%.2f")
DIS = st.number_input("Distance to Employment Centers", min_value=0.0, format="%.2f")
RAD = st.number_input("Accessibility to Highways", min_value=1, max_value=24, step=1)
TAX = st.number_input("Property Tax Rate", min_value=0, step=1)
PTRATIO = st.number_input("Pupil-Teacher Ratio", min_value=0.0, format="%.2f")
B = st.number_input("Proportion of Black Residents", min_value=0.0, format="%.2f")
LSTAT = st.number_input("Lower Status Population (%)", min_value=0.0, format="%.2f")

# Convert input into a NumPy array
features = np.array([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]])

# Predict button
if st.button("Predict Price"):
    prediction = model.predict(features)
    st.success(f"🏠 Estimated House Price: ${prediction[0]*1000:.2f}")
