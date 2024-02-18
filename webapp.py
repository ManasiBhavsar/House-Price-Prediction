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

# Load the dataset
#@st.cache
def load_data():
    file_path = r'D:/House Price Prediction/Housing.csv'
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        print("Error: CSV file not found")
        return None

#@st.cache_data
def load_model():
    try :
        model=pickle.load(open('rf_mofel.pkl','rb'))
        
        print("Model loaded successfully")
        print("Model type:", type(model))
        return model
    except FileNotFoundError:
        print("Error: Model file not found")
    except Exception as e:
        print("Error loading model:", e)
    return None



#from tkinter import filedialog
#import tkinter as tk

# Create a Tkinter root window
#root = tk.Tk()
#root.withdraw()  # Hide the root window

# Open a file dialog box to select the encoder file
#encoder_path = filedialog.askopenfilename(title="Select Encoder File", filetypes=[("Pickle files", "*.pkl")])

# Load the encoder object
#with open(encoder_path, 'rb') as f:
#    encoder = pickle.load(f)

# Now you can use the encoder object as needed


# Function to preprocess input data
#def preprocess_input(area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus):
    # Encode categorical features
#    mainroad_encoded = encoder.transform(np.array([mainroad]).reshape(-1, 1))
 ##  basement_encoded = encoder.transform(np.array([basement]).reshape(-1, 1))
   # hotwaterheating_encoded = encoder.transform(np.array([hotwaterheating]).reshape(-1, 1))
    #airconditioning_encoded = encoder.transform(np.array([airconditioning]).reshape(-1, 1))
#    parking_encoded = encoder.transform(np.array([parking]).reshape(-1, 1))
#    prefarea_encoded = encoder.transform(np.array([prefarea]).reshape(-1, 1))
#    furnishingstatus_encoded = encoder.transform(np.array([furnishingstatus]).reshape(-1, 1))
#    

#    # Combine all features into a single array
#    features = np.array([area, bedrooms, bathrooms, stories]).reshape(1, -1)
#    features = np.concatenate([features, mainroad_encoded, guestroom_encoded, basement_encoded,
#                               hotwaterheating_encoded, airconditioning_encoded, parking_encoded,
#                               prefarea_encoded, furnishingstatus_encoded], axis=1)
#    return features


#model = pickle.load(open(r'‪D:/Downloads/trained_model.sav', 'rb'))

def predict_price(input_data,model):
    """
    Function to predict house price based on the input features.
    """
    features = np.array([input_data]).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]

# Streamlit app
def main():
    st.title("House Price Prediction App")
    st.write("Select values for each feature to predict the house price.")

    model=load_model()

    # Dropdown menus for each column in the dataset
    area = st.number_input("Area (in square feet)")
    bedrooms = st.number_input("Number of bedrooms", min_value=1, max_value=10, step=1)
    bathrooms = st.number_input("Number of bathrooms", min_value=1, max_value=10, step=1)
    stories = st.number_input("Number of stories", min_value=1, max_value=10, step=1)
    mainroad = st.selectbox("Main road", [1, 0])
    guestroom = st.selectbox("Guest room", [1, 0])
    basement = st.selectbox("Basement", [1, 0])
    hotwaterheating = st.selectbox("Hot water heating", [1, 0])
    airconditioning = st.selectbox("Air conditioning", [1, 0])
    parking = st.number_input("Parking", min_value=0,max_value=4,step=1) 
    prefarea = st.selectbox("Preferred area", [1, 0])
    furnishingstatus = st.selectbox("Furnishing status", [0,1,2])
    
    
    input_data =(area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating,
                                airconditioning, parking, prefarea, furnishingstatus)
    
    input_data=np.array(input_data).reshape(1,-1)    
    st.write(input_data)
    

    # Predict button
    if st.button("Predict"):
        # Make prediction
        prediction = model.predict(input_data)
        st.success(f"The estimated price of the house is ${prediction[0]:,.2f}")
        
if __name__ == "__main__":
    main()
