import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from model_utils import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    MODEL_DIR,
    MODEL_PATH,
    METADATA_PATH,
    NUMERIC_FEATURES,
    ROOT_DIR,
)


DATA_PATH = ROOT_DIR / "raw_data" / "G3.csv"
TARGET_COLUMN = "revenue"


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Training data not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in data]
    if missing_columns:
        raise ValueError(f"Training data is missing columns: {', '.join(missing_columns)}")

    training_data = data.loc[
        data[TARGET_COLUMN].notna() & (data[TARGET_COLUMN] >= 0)
    ].copy()
    if training_data.empty:
        raise ValueError("No non-negative revenue values are available for training.")

    x_train = training_data[FEATURE_COLUMNS].copy()
    for column in CATEGORICAL_FEATURES:
        x_train[column] = x_train[column].fillna("").astype(str)

    y_train = np.log1p(training_data[TARGET_COLUMN].astype(float))

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                GradientBoostingRegressor(
                    n_estimators=150,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(x_train, y_train)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    currencies = sorted(
        training_data["revenue_currency"].dropna().astype(str).unique().tolist()
    )
    if not currencies:
        raise ValueError("No revenue currencies are available in the training data.")
    metadata = {
        "target": TARGET_COLUMN,
        "target_transform": "log1p",
        "training_rows": int(len(training_data)),
        "currencies": currencies,
    }
    with METADATA_PATH.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    print(f"Trained on {len(training_data)} rows.")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Available revenue currencies: {', '.join(currencies)}")


if __name__ == "__main__":
    main()
