import pandas as pd
import yaml
import pickle
import os
import sys

# Is line se 'src' folder import ke liye enable ho jata hai
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import get_preprocessor

def train_model():
    # Load Config
    with open('../config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load Data
    df = pd.read_csv('../' + config['data']['raw_path'])
    
    X = df.drop(columns=config['features']['drop_columns'])
    y = df[config['features']['target']]
    
    # Get Preprocessor
    preprocessor = get_preprocessor(config['features']['numeric'], config['features']['categorical'])
    
    # Model
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline
    
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('regressor', RandomForestRegressor(n_estimators=100))])
    
    clf.fit(X, y)
    
    # Save Model
    with open('../models/model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    print("✅ Model Trained and Saved!")

if __name__ == "__main__":
    train_model()