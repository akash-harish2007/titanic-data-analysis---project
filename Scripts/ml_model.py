import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

os.makedirs('Data/outputs', exist_ok=True)

print(" Loading cleaned data...")

# Load data
train = pd.read_csv('Data/Raw/processed/titanic_cleaned.csv')
test = pd.read_csv('Data/Raw/processed/titanic_test_cleaned.csv')

print(f" Training data: {train.shape}")
print(f" Test data: {test.shape}")

# Separate features and target
if 'Survived' in train.columns:
    X_train = train.drop('Survived', axis=1)
    y_train = train['Survived']
else:
    print(" 'Survived' not found in train data!")
    exit()

# Handle test data
if 'Survived' in test.columns:
    X_test = test.drop('Survived', axis=1)
    y_test = test['Survived']
    has_test_labels = True
else:
    if 'passenger_id' in test.columns:
        passenger_ids = test['passenger_id']
        X_test = test.drop('passenger_id', axis=1)
    else:
        passenger_ids = None
        X_test = test
    y_test = None
    has_test_labels = False

print(f"\n Original X_train shape: {X_train.shape}")
print(f" Original X_test shape: {X_test.shape}")


print("\n Dropping text columns...")
columns_to_drop = ['Cabin', 'Name', 'Ticket']
X_train = X_train.drop(columns=columns_to_drop, errors='ignore')
X_test = X_test.drop(columns=columns_to_drop, errors='ignore')
print(f"    Dropped: {columns_to_drop}")


print("\n Encoding categorical columns...")

# Find non-numeric columns in BOTH train and test
non_numeric_train = X_train.select_dtypes(include=['object']).columns.tolist()
non_numeric_test = X_test.select_dtypes(include=['object']).columns.tolist()
all_non_numeric = list(set(non_numeric_train + non_numeric_test))

print(f"   Non-numeric columns: {all_non_numeric}")

# Encode each non-numeric column
for col in all_non_numeric:
    le = LabelEncoder()
    # Combine train and test to handle all categories
    if col in X_train.columns and col in X_test.columns:
        combined = pd.concat([X_train[col], X_test[col]]).astype(str)
        le.fit(combined)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        print(f"    Encoded: {col}")
    elif col in X_train.columns:
        le.fit(X_train[col].astype(str))
        X_train[col] = le.transform(X_train[col].astype(str))
        print(f"    Encoded (train only): {col}")


print("\n Handling missing values...")
X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_train.median())  # Use training median
print("    Filled missing values")


print("\n Aligning columns...")

# Get the intersection of columns (common columns in both)
common_columns = list(set(X_train.columns) & set(X_test.columns))
print(f"   Common columns: {len(common_columns)}")

# Keep only common columns in BOTH
X_train = X_train[common_columns]
X_test = X_test[common_columns]

# Reorder test columns to match train
X_test = X_test[X_train.columns]

print(f"\n Final X_train shape: {X_train.shape}")
print(f" Final X_test shape: {X_test.shape}")
print(f"   Columns match: {list(X_train.columns) == list(X_test.columns)}")


print("\n🤖 Training Random Forest Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"\n Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.4f}")


y_pred = model.predict(X_test)


if has_test_labels:
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    print(f"\n Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\n Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred))


feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n Top 10 Features:")
print(feature_importance.head(10))

# Visualize feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance.head(10)['feature'], 
         feature_importance.head(10)['importance'])
plt.xlabel('Importance')
plt.title('Top 10 Features for Survival Prediction')
plt.tight_layout()
plt.savefig('Data/outputs/feature_importance.png')
plt.show()


if not has_test_labels and passenger_ids is not None:
    submission = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': y_pred
    })
    submission.to_csv('Data/outputs/submission.csv', index=False)
    print("\n Submission saved to Data/outputs/submission.csv")
    print(submission.head())

# Save results
feature_importance.to_csv('Data/outputs/feature_importance.csv', index=False)
joblib.dump(model, 'Data/outputs/titanic_model.pkl')
print("\n Model saved to Data/outputs/titanic_model.pkl")

print("\n🎉 All done!")