# app.py
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model and preprocessor
try:
    with open('src/trained_crop_model/dtr.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('src/trained_crop_model/preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
except FileNotFoundError:
    st.error("Model or preprocessor files not found. Make sure 'dtr.pkl' and 'preprocessor.pkl' are in the 'trained_crop_model' directory.")
    st.stop()

# --- MODIFICATION START ---
# Load the dataframe to get unique values for dropdowns
try:
    df = pd.read_csv('src/yield_df.csv')
    # Drop duplicates and nulls to ensure clean lists, similar to your notebook
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    
    countries = sorted(df['Area'].unique())
    crops = sorted(df['Item'].unique())
except FileNotFoundError:
    st.error("The 'yield_df.csv' file was not found. Please add it to the project directory.")
    # As a fallback, you can use hardcoded lists, but it's not ideal
    countries = ['India', 'USA', 'Brazil'] # Example fallback
    crops = ['Maize', 'Potatoes', 'Wheat'] # Example fallback
# --- MODIFICATION END ---


# Set up the title of the web app
st.title('🌾 Crop Yield Prediction')
st.write("Enter the details below to predict the crop yield (in hg/ha).")


# Create columns for a better layout
col1, col2 = st.columns(2)

with col1:
    # Input fields for the user
    year = st.number_input('Year', min_value=1990, max_value=2030, value=2010)
    avg_rain = st.number_input('Average Rainfall (mm/year)', value=70.0, format="%.2f")
    pesticides = st.number_input('Pesticides (tonnes)', value=3500.0, format="%.2f")

with col2:
    area = st.selectbox('Country', countries)
    avg_temp = st.number_input('Average Temperature (°C)', value=28.0, format="%.2f")
    item = st.selectbox('Crop', crops)


# Prediction button
if st.button('Predict Yield', use_container_width=True):
    try:
        # Prepare the feature array in the correct order for the preprocessor
        # Order: Year, Area, average_rain_fall_mm_per_year, avg_temp, pesticides_tonnes, Item
        features = np.array([[year, area, avg_rain, avg_temp, pesticides, item]])

        # Transform the features using the loaded preprocessor
        transformed_features = preprocessor.transform(features)

        # Predict using the loaded model
        prediction = model.predict(transformed_features).reshape(1, -1)

        # Display the result
        st.success(f'**Predicted Yield: {prediction[0][0]:.2f} hg/ha**')

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")