import pandas as pd
import yaml
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from preprocessing import get_preprocessor

def train_optimized_model():
    # --- Dynamic Path Resolution ---
    # Isse hum folder ki exact location nikal rahe hain
    current_dir = os.path.dirname(os.path.abspath(__file__)) # 'src' folder
    root_dir = os.path.dirname(current_dir) # Project ka main folder
    
    config_path = os.path.join(root_dir, 'config', 'config.yaml')
    
    # 1. Load Config
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Error: Config file not found at {config_path}")
        return

    # 2. Data Path Setup
    # config['data']['raw_path'] hamesha root se calculate hoga
    data_path = os.path.join(root_dir, config['data']['raw_path'])
    
    # 3. Load Data
    if not os.path.exists(data_path):
        print(f"❌ Error: CSV file not found at {data_path}")
        return
        
    df = pd.read_csv(data_path)
    X = df.drop(columns=config['features']['drop_columns'])
    y = df[config['features']['target']]
    
    # 4. Preprocessing
    preprocessor = get_preprocessor(config['features']['numeric'], config['features']['categorical'])
    
    # 5. Define Pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])
    
    # 6. Hyperparameter Grid
    param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [None, 10],
        'regressor__min_samples_split': [2, 5]
    }
    
    # 7. Grid Search
    print("🔍 Tuning and Training started...")
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['data']['test_size'], random_state=42)
    grid_search.fit(X_train, y_train)
    
    print(f"✅ Best R2 Score: {grid_search.best_score_:.4f}")
    
    # 8. Save Model
    model_save_path = os.path.join(root_dir, config['model']['model_save_path'])
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    with open(model_save_path, 'wb') as f:
        pickle.dump(grid_search.best_estimator_, f)
        
    print(f"💾 Optimized Model Saved at: {model_save_path}")

if __name__ == "__main__":
    train_optimized_model()