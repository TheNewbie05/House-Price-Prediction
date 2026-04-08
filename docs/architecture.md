# Project Architecture

1. **Data Layer**: CSV files stored in `/data`.
2. **Logic Layer**: Preprocessing and model training handled in `/src`.
3. **Model Layer**: Serialization using Pickle in `/models`.
4. **Service Layer**: 
   - UI: Streamlit Web App (`app/main.py`)
   - API: FastAPI Backend (`app/api.py`)