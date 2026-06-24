import pandas as pd

# Load test data
df = pd.read_csv('Data/Raw/test.csv')

# Check missing values BEFORE
print("Before cleaning:")
print(df.isnull().sum())
print()

# --- FILL MISSING VALUES ---

# 1. Age - fill with median (use SAME median as training - 28.0)
median_age = 28.0  # From training data
df['Age'] = df['Age'].fillna(median_age)
print(f"✅ Age fixed (median: {median_age})")

# 2. Embarked - fill with mode (use SAME mode as training - 'S')
mode_embarked = 'S'  # From training data
df['Embarked'] = df['Embarked'].fillna(mode_embarked)
print(f"✅ Embarked fixed (mode: {mode_embarked})")

# 3. Fare - fill with median (use SAME median as training - 14.45)
median_fare = 14.45  # From training data
df['Fare'] = df['Fare'].fillna(median_fare)
print(f"✅ Fare fixed (median: {median_fare})")

# 4. Cabin - drop the column (SAME as training)
df = df.drop('Cabin', axis=1)
print("✅ Cabin dropped")

# Check missing values AFTER
print("\nAfter cleaning:")
print(df.isnull().sum())
print()

# Check specific columns
print(f"Age missing count: {df['Age'].isnull().sum()}")  # Should be 0
print(f"Fare missing count: {df['Fare'].isnull().sum()}")  # Should be 0
print(f"Embarked missing count: {df['Embarked'].isnull().sum()}")  # Should be 0

# --- CONVERT CATEGORICAL TO NUMERIC (SAME as training) ---

# Replacing categorical data of sex to numerical
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
print("✅ Sex converted")

# Converting embarked to numeric
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
print("✅ Embarked converted")

# --- CREATE NEW COLUMNS (SAME as training) ---

# Creating age group columns
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 19, 40, 60, 100],
                        labels=['Child', 'Teen', 'Adult', 'Middle-aged', 'Senior'])
print("✅ AgeGroup created")

# Adding sibsp and parch column as family_size column
df['Family_Size'] = df['SibSp'] + df['Parch']
print("✅ Family_Size created")

# Splitting fare column into four categories
df['fare_Group'] = pd.cut(df['Fare'], bins=[0, 50, 100, 200, 600],
                          labels=['Very-Low', 'Low', 'Medium', 'High'])
print("✅ fare_Group created")

# --- SAVE CLEANED TEST DATA ---

# Save cleaned test data
df.to_csv('Data/Raw/processed/titanic_test_cleaned.csv', index=False)
print("✅ Cleaned test data saved to data/processed/titanic_test_cleaned.csv")

# --- VERIFICATION ---
print("\n" + "="*50)
print("✅ TEST DATA CLEANING COMPLETE!")
print("="*50)

print(f"\n📊 Test Data Shape: {df.shape}")
print(f"📊 Columns: {df.columns.tolist()}")

print("\n📊 First 5 rows:")
print(df.head())

print("\n🔍 Missing Values Check:")
print(df.isnull().sum())  