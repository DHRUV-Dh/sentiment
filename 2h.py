import streamlit as st
import requests

st.title("Text Sentiment Predictor")

prediction_endpoint = "http://127.0.0.1:5000/predict"

user_input = st.text_input("Enter text and click on Predict", "")

if st.button("Predict") and user_input:
    response = requests.post(prediction_endpoint, json={"text": user_input})
    st.write("Response content:", response.content)  # Print response content
    try:
        response_json = response.json()
        st.write(f"Predicted sentiment: {response_json['prediction']}")
    except Exception as e:
        st.error(f"Error decoding JSON: {e}")
