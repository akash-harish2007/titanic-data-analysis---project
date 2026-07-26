import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from sklearn.model_selection import cross_val_score

import joblib
import os


from data_preprocessing import clean_data



os.makedirs(
    "Data/outputs",
    exist_ok=True
)



# Get processed data

X_train, X_test, y_train, y_test = clean_data()



print("\nTraining model...")



model = RandomForestClassifier(

    n_estimators=500,
    max_depth=8,
    min_samples_split=5,
    random_state=42
)



model.fit(

    X_train,

    y_train
)



print("\nMaking predictions...")


y_pred = model.predict(
    X_test
)



# Evaluation


accuracy = accuracy_score(

    y_test,

    y_pred

)


print(
    f"\nAccuracy: {accuracy:.4f}"
)



print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)



print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)



# Cross validation


scores = cross_val_score(

    model,

    X_train,

    y_train,

    cv=5

)


print(
    "\nCV Scores:",
    scores
)


print(
    "Mean CV:",
    scores.mean()
)



# Save model


joblib.dump(

    model,

    "Data/outputs/titanic_model.pkl"

)


print(
    "\nModel saved!"
)