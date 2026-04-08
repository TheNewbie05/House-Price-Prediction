from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
import os

# Create FastAPI app
app = FastAPI(title="House Price Prediction API", version="1.0")

# --- Path Logic ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.pkl')

# Load Model
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Define Input Schema (Validation)
class HouseInput(BaseModel):
    Area: float
    Bedrooms: int
    Bathrooms: int
    Age: int
    Location: str
    Property_Type: str

@app.get("/")
def read_root():
    return {"status": "Online", "message": "Welcome to the House Price API"}

@app.post("/predict")
def predict(data: HouseInput):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Predict
    prediction = model.predict(input_df)[0]
    
    return {
        "predicted_price": round(float(prediction), 2),
        "currency": "INR"
    }