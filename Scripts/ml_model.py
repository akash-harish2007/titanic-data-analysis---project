import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

# Create outputs folder
os.makedirs('Data/outputs', exist_ok=True)

print("📂 Loading cleaned data...")

# Load cleaned data
train = pd.read_csv('Data/Raw/processed/titanic_cleaned.csv')
test = pd.read_csv('Data/Raw/processed/titanic_test_cleaned.csv')

print(f"✅ Training data: {train.shape}")
print(f"✅ Test data: {test.shape}")

print("\n📊 Training columns:")
print(train.columns.tolist())