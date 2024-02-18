# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 23:27:32 2024

@author: manas
"""

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
import requests
import sklearn

model_url='https://raw.githubusercontent.com/ManasiBhavsar/House-Price-Prediction/blob/main/finalized_model.sav'
responsee=requests.get(model_url)

if responsee.status_code==200:
    with open('finalized_model.sav','wb') as f:
        f.write(responsee.content)
else:
    print("Failed to download the model file")
if os.path.exists('finalized_model.sav'):
    with open('finalized_model.sav', 'rb') as f:
        loaded_model = pickle.load(open("finalized_model.sav",'rb'))
else:
    st.error("Model file not found")

# Load the model
#with open('finalized_model.sav', 'rb') as f:
    #loaded_model = pickle.load(f)
#loaded_model = responsee
#load model
#def load_model():
    #try :
        
        #print("Model loaded successfully")
        #print("Model type:", type(model))
        #return model
    #except FileNotFoundError:
     #   print("Error: Model file not found")
    #except Exception as e:
     #   print("Error loading model:", e)
    #return None




def predict_price(input_data):
    """
    Function to predict house price based on the input features.
    """
    features = np.array([input_data]).reshape(1, -1)
    prediction = loaded_model.predict(features)
    return prediction[0]

# Streamlit app
def main():
    st.title("House Price Prediction App")
    st.write("Select values for each feature to predict the house price.")

    #model=pickle.load(open('https://github.com/ManasiBhavsar/House-Price-Prediction/blob/main/trained_model.sav','rb'))


    # Dropdown menus for each column in the dataset
    area = st.number_input("Area (in square feet)",min_value=1600,max_value=16000)
    bedrooms = st.number_input("Number of bedrooms", min_value=1, max_value=10, step=1)
    bathrooms = st.number_input("Number of bathrooms", min_value=1, max_value=10, step=1)
    stories = st.number_input("Number of stories", min_value=1, max_value=10, step=1)
    mainroad = st.selectbox("Main road (Yes:1 , No:0)", [1, 0])
    guestroom = st.selectbox("Guest room (Yes:1 , No:0)", [1, 0])
    basement = st.selectbox("Basement (Yes:1 , No:0)", [1, 0])
    hotwaterheating = st.selectbox("Hot water heating (Yes:1 , No:0)", [1, 0])
    airconditioning = st.selectbox("Air conditioning (Yes:1 , No:0)", [1, 0])
    parking = st.number_input("Parking", min_value=0,max_value=4,step=1) 
    prefarea = st.selectbox("Preferred area (Yes:1 , No:0)", [1, 0])
    furnishingstatus = st.selectbox("Furnishing status (Unfurnished:0 , Semi-furnished:1 , Furnished:2)", [0,1,2])
    
    
    input_data =(area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating,
                                airconditioning, parking, prefarea, furnishingstatus)
    
    #input_data = (np.array(input_data)).reshape(1,-1)
    
    # Predict button
    if st.button("Predict"):
        # Make prediction
        prediction = predict_price(input_data)
        st.success(f"The estimated price of the house is ${prediction[0]:,.2f}")
        
if __name__ == "__main__":
    main()
