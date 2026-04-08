import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set Plot style
sns.set_theme(style="whitegrid")

# Dynamic Path Setup
# This finds the data folder regardless of where you run the notebook from
BASE_DIR = os.path.dirname(os.getcwd())
DATA_PATH = os.path.join(BASE_DIR, 'data', 'house_prices.csv')

# Load Dataset
print(f"Loading data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# --- 1. Basic Inspection ---
print("--- Dataset Overview ---")
print(df.info())
print("\n--- Summary Statistics ---")
print(df.describe())

# --- 2. Target Variable Analysis (Price) ---
plt.figure(figsize=(10, 6))
sns.histplot(df['Price'], kde=True, color='teal')
plt.title('Distribution of House Prices', fontsize=15)
plt.xlabel('Price (INR)')
plt.ylabel('Frequency')
plt.show()

# --- 3. Feature Correlation ---
# Checking which factors (Area, Bedrooms, etc.) influence the Price most
plt.figure(figsize=(12, 8))
numeric_df = df.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numeric Features', fontsize=15)
plt.show()

# --- 4. Categorical Analysis (Location vs Price) ---
plt.figure(figsize=(10, 6))
sns.boxplot(x='Location', y='Price', data=df, palette='viridis')
plt.title('House Prices by Location', fontsize=15)
plt.show()