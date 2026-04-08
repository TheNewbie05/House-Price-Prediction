import streamlit as st
import pandas as pd
import pickle
import os

# Styling
st.set_page_config(page_title="Real Estate AI", page_icon="🏘️", layout="centered")

# --- Path Logic ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.pkl')

@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

model = load_model()

# UI
st.title("🏡 House Price Predictor")
st.write("Enter the property details below to get an instant valuation.")

with st.container(border=True):
    col1, col2 = st.columns(2)
    with col1:
        area = st.number_input("Area (sq ft)", value=1500, step=100)
        loc = st.selectbox("Location", ["City Center", "Suburb", "Rural"])
        p_type = st.selectbox("Property Type", ["Apartment", "House", "Villa"])
    with col2:
        beds = st.slider("Bedrooms", 1, 5, 2)
        baths = st.slider("Bathrooms", 1, 3, 1)
        age = st.number_input("Age of Property (Years)", 0, 50, 5)

if st.button("Calculate Estimated Price", use_container_width=True):
    input_data = pd.DataFrame([[area, beds, baths, age, loc, p_type]], 
                             columns=['Area', 'Bedrooms', 'Bathrooms', 'Age', 'Location', 'Property_Type'])
    
    res = model.predict(input_data)[0]
    st.success(f"### Predicted Price: ₹ {res:,.2f}")
    st.info(f"Price per sq. ft.: ₹ {res/area:,.2f}")