#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import pickle
import base64

# Load the model
def load_model(path):
    with open(path, 'rb') as file:
        model = pickle.load(file)
    return model

# Make prediction using the model
def predict(model, data):
    # Drop the 'ID' column before prediction
    data_cleaned = data.drop(columns=["ID"], errors="ignore")
    
    # Align the columns of input data with the trained model's feature columns
    with open(r"C:\Users\bathl\OneDrive\Documents\SJSU\Machine_Learning\Project\ML_Project\feature_names.pkl", 'rb') as file:
        feature_names = pickle.load(file)
    
    # Ensure the input data has the same columns as the model's training data
    data_cleaned = data_cleaned[feature_names]
    
    # Perform the prediction
    prediction_numeric = model.predict(data_cleaned)
    
    # Class mapping
    class_mapping = {
        0: "Ground",
        1: "Parking",
        2: "Rooftop"
    }
    
    # Map numeric predictions to their corresponding class names
    prediction = [class_mapping[p] for p in prediction_numeric]
    
    return prediction

# Convert image file to base64
def get_base64_of_file(file_path):
    with open(file_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Load trained RandomForest model
model = load_model(r"C:\Users\bathl\OneDrive\Documents\SJSU\Machine_Learning\Project\ML_Project\RandomeForest.pkl")

# Background image in Streamlit
image_path = r"C:\Users\bathl\Downloads\sp13.jpg"
base64_image = get_base64_of_file(image_path)

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{base64_image}");
        background-size: cover;
        background-position: center center;
        font-family: "Verdana", sans-serif;
    }}
    .stFileUploader .st-az, .stFileUploader .st-ay {{
        color: #FFDD00 !important; /* Bright Yellow */
    }}
    * {{
        color: #1F77B4 !important; /* Vibrant Blue */
    }}
    .dataframe, .dataframe th, .dataframe td {{
        background-color: rgba(255, 255, 255, 0.85) !important; /* Light Transparent White */
        color: #000000 !important; /* Black Text */
    }}
    .stMetric {{
        background-color: #FFFFFF !important; /* White */
        color: #FFFFFF !important; /* White Text */
        border-radius: 10px;
        padding: 10px;
        font-weight: bold;
    }}
    h1, h2, h3 {{
        color: #00A86B !important; /* Bright Green */
    }}
    .stButton>button {{
        background-color: #FF4500 !important; /* Bright Orange */
        color: #FFFFFF !important; /* White Text */
        border: None;
        border-radius: 8px;
        font-weight: bold;
        font-size: 16px;
    }}
    .stButton>button:hover {{
        background-color: #FF6347 !important; /* Light Orange */
        color: #000000 !important; /* Black Text */
    }}
    </style>
    """,
    unsafe_allow_html=True
)


st.title('Predicting Solar Panel Installations')

with st.container():
    uploaded_file = st.file_uploader("Choose a data file", type=['csv'], key='1')
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if not df.empty:
            st.success("Data File Uploaded Successfully!")
            
            st.markdown("### Preview of the data used for prediction:")
            st.dataframe(df.iloc[0:1].style.set_properties(**{'background-color': 'black', 'color': 'yellow'}))
                
            # Make prediction
            prediction = predict(model, df.iloc[0:1])
            st.metric(label="Solar Installation Prediction", value=f"{prediction[0]}")
        else:
            st.error("The uploaded data file is empty.")
    else:
        st.info("Please upload a data file to proceed.")

