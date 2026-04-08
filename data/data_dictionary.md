# 🏗️ Project Architecture

This project follows a modular and production-ready architecture for Machine Learning.

## 1. Data Layer
- **Source**: `data/house_prices.csv`
- **Storage**: Raw data is stored locally. Data dictionary defines the schema.

## 2. Logic Layer (Source Code)
- **Preprocessing**: `src/preprocessing.py` contains the Scikit-Learn Pipeline for scaling and encoding.
- **Training**: `src/train_model.py` performs Hyperparameter tuning using GridSearchCV.

## 3. Model Layer
- **Serialization**: The best performing model is saved as `models/model.pkl` using Pickle.

## 4. Service Layer
- **Web App**: Built with **Streamlit** for human interaction.
- **API**: Built with **FastAPI** for machine-to-machine communication.

## 5. Quality Assurance
- **Tests**: Unit tests in `tests/` ensure model loading and basic functionality.