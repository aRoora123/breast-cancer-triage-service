
import pandas as pd
from sklearn.datasets import load_breast_cancer


def main():
    data = load_breast_cancer()

    # Store the 30 features in a DataFrame
    df = pd.DataFrame(data.data, columns=data.feature_names)

    # Make column names easier to use in the API
    df.columns = df.columns.str.replace(" ", "_")

    # Add the target column
    df["target"] = data.target

    # Save the dataset
    df.to_csv("data/raw/breast_cancer.csv", index=False)

    print("Rows:", len(df))
    print("Features:", len(df.columns) - 1)
    print("Missing values:", df.isnull().sum().sum())
    print("Dataset saved successfully.")


if __name__ == "__main__":
    main()
