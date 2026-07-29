import streamlit as st
import joblib
import numpy as np

model = joblib.load("house_price_model.pkl")

st.title("🏠 House Price Prediction")

area = st.number_input("sqft_living")
bedrooms = st.number_input("bedrooms")
bathrooms = st.number_input("bathrooms")
floors = st.number_input("floors")
sqft_living15 = st.number_input("sqft_living15")

if st.button("Predict Price"):
    features = np.array([[area, bedrooms, bathrooms, floors, sqft_living15]])
    prediction = model.predict(features)

    st.success(f"Predicted Price: ₹ {prediction[0]:,.2f}")