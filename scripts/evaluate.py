import pickle
import pandas as pd
import os
from sklearn.metrics import r2_score

def evaluate():
    # Paths setup
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(root, 'models', 'model.pkl')
    data_path = os.path.join(root, 'data', 'house_prices.csv')
    
    # Load
    model = pickle.load(open(model_path, 'rb'))
    df = pd.read_csv(data_path)
    
    # Simple evaluation on full data
    X = df.drop(['Price', 'Property_ID'], axis=1)
    y = df['Price']
    
    predictions = model.predict(X)
    score = r2_score(y, predictions)
    print(f"Current Model R2 Score: {score:.4f}")

if __name__ == "__main__":
    evaluate()