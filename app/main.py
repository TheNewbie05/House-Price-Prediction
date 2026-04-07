import streamlit as st
import pandas as pd
import pickle
import os

# Page styling
st.set_page_config(page_title="House Price Predictor", page_icon="🏡")

def load_model():
    # Model ka path set karein (src se ek level upar jaakar models folder mein)
    model_path = os.path.join(os.path.dirname(__file__), '../models/model.pkl')
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def main():
    st.title("🏡 House Price Prediction App")
    st.markdown("Wanna Know the House Price by Prediction, Here is the Price Calculator.")
    
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Model is not able to load . first run the 'src/train_model.py' file then  generate model. Error: {e}")
        return

    # User Input Form
    with st.form("prediction_form"):
        st.subheader("Fill details about house:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            area = st.number_input("Area (sq ft)", min_value=500, max_value=10000, value=2500, step=50)
            bedrooms = st.slider("Bedrooms", 1, 5, 3)
            bathrooms = st.slider("Bathrooms", 1, 3, 2)
            
        with col2:
            age = st.number_input("Property's Age (Years)", min_value=0, max_value=50, value=5)
            location = st.selectbox("Location", ["City Center", "Suburb", "Rural"])
            property_type = st.selectbox("Property Type", ["Apartment", "House", "Villa"])
            
        submit_button = st.form_submit_button("Calculate")

    if submit_button:
        # Input data ko DataFrame mein convert karein (Features ka order wahi hona chahiye jo training mein tha)
        input_data = pd.DataFrame([{
            'Area': area,
            'Bedrooms': bedrooms,
            'Bathrooms': bathrooms,
            'Age': age,
            'Location': location,
            'Property_Type': property_type
        }])
        
        # Prediction
        prediction = model.predict(input_data)[0]
        
        # Display Result
        st.success(f"### Estimated Market Price: ₹ {prediction:,.2f}")
        st.balloons()

if __name__ == "__main__":
    main()