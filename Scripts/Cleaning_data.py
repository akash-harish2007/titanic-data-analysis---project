import pandas as pd

# Load data
df = pd.read_csv('Data/Raw/train.csv')

# Check missing values BEFORE
print("Before cleaning:")
print(df.isnull().sum())
print()

# --- FILL MISSING VALUES ---

# 1. Age - fill with median 
median_age = df['Age'].median()
df['Age'] = df['Age'].fillna(median_age)
print(f"✅ Age fixed (median: {median_age})")

# 2. Embarked - fill with mode
mode_embarked = df['Embarked'].mode()[0]
df['Embarked'] = df['Embarked'].fillna(mode_embarked)
print(f"✅ Embarked fixed (mode: {mode_embarked})")

# 3. Cabin - drop the column 
df = df.drop('Cabin', axis=1)
print("✅ Cabin dropped")

# Check missing values AFTER
print("\nAfter cleaning:")
print(df.isnull().sum())
print()

# Check Age specifically
print(f"Age missing count: {df['Age'].isnull().sum()}")  # Should be 0
print(f"Age sample: {df['Age'].head()}")

#replacing categorical data of sex to numerical 
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
print(df['Sex'])

#converting embarked to numeric
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

#creating age group columns
df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 19, 40, 60, 100] ,
                        labels=['Child', 'Teen', 'Adult', 'Middle-aged', 'Senior'])
print(df.head())

#adding sibsp and parch column as family_size column

df['Family_Size'] = df['SibSp'] + df['Parch']
print(df['Family_Size'])

#splitting fare column into four categories
df['fare_Group'] = pd.cut(df['Fare'], bins=[0,50,100,200,250] , 
                          labels=['Very-Low', 'Low', 'Medium', 'High'])
print(df.head())

# Save cleaned training data
df.to_csv('Data/Raw/processed/titanic_cleaned.csv', index=False)
print("✅ Cleaned training data saved to data/processed/titanic_cleaned.csv")