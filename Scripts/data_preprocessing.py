import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def clean_data():

    print("Loading data...")


    df = pd.read_csv(
        "Data/Raw/processed/titanic_cleaned.csv"
    )


    print("Original shape:", df.shape)



    # Remove unnecessary columns

    columns_to_drop = [
        "Cabin",
        "Name",
        "Ticket"
    ]


    df = df.drop(
        columns=columns_to_drop,
        errors="ignore"
    )



    # Separate X and y

    X = df.drop(
        "Survived",
        axis=1
    )


    y = df["Survived"]



    # Find categorical columns

    categorical_columns = X.select_dtypes(
        include="object"
    ).columns



    # Encode categorical values

    encoder = LabelEncoder()


    for column in categorical_columns:

        X[column] = encoder.fit_transform(
            X[column].astype(str)
        )



    # Fill missing values

    X = X.fillna(
        X.median()
    )



    # Split dataset

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.2,

        random_state=42
    )



    print("Training data:", X_train.shape)

    print("Testing data:", X_test.shape)



    return (
        X_train,
        X_test,
        y_train,
        y_test
    )