import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def clean_data():

    print("Loading data...")


    df = pd.read_csv(
        "Data/Raw/processed/titanic_cleaned.csv"
    )


    print("Original shape:", df.shape)


    # =========================
    # Feature Engineering
    # =========================

    # Extract Title from Name

    df["Title"] = df["Name"].str.extract(
        r',\s*([^\.]+)\.'
    )


    # Create Family Size

    df["FamilySize"] = (
        df["SibSp"]
        +
        df["Parch"]
        +
        1
    )


    # Create IsAlone

    df["IsAlone"] = (
        df["FamilySize"] == 1
    ).astype(int)



    # =========================
    # Remove unnecessary columns
    # =========================

    columns_to_drop = [
        "Cabin",
        "Name",
        "Ticket"
    ]


    df = df.drop(
        columns=columns_to_drop,
        errors="ignore"
    )



    # =========================
    # Separate Target
    # =========================

    y = df["Survived"]


    df = df.drop(
        "Survived",
        axis=1
    )



    # =========================
    # Find categorical columns
    # =========================

    categorical_columns = df.select_dtypes(
        include="object"
    ).columns


    numerical_columns = df.select_dtypes(
        exclude="object"
    ).columns



    print(
        "Categorical columns:",
        list(categorical_columns)
    )



    # =========================
    # One Hot Encoding
    # =========================

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )


    encoded = encoder.fit_transform(
        df[categorical_columns]
    )


    encoded_columns = encoder.get_feature_names_out(
        categorical_columns
    )


    df_encoded = pd.DataFrame(
        encoded,
        columns=encoded_columns,
        index=df.index
    )



    # Keep numerical columns + encoded columns

    df_final = pd.concat(
        [
            df[numerical_columns],
            df_encoded
        ],
        axis=1
    )



    # =========================
    # Handle Missing Values
    # =========================

    df_final = df_final.fillna(
        df_final.median()
    )



    print(
        "Final shape:",
        df_final.shape
    )



    # =========================
    # Train Test Split
    # =========================

    X_train, X_test, y_train, y_test = train_test_split(

        df_final,

        y,

        test_size=0.2,

        random_state=42
    )



    print(
        "Training data:",
        X_train.shape
    )


    print(
        "Testing data:",
        X_test.shape
    )



    return (
        X_train,
        X_test,
        y_train,
        y_test
    )