import math

import joblib
import pandas as pd
from flask import Flask, redirect, render_template, request, url_for

from model_utils import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    METADATA_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
    load_metadata,
)


app = Flask(__name__)

if not MODEL_PATH.is_file() or not METADATA_PATH.is_file():
    raise RuntimeError(
        "Revenue model files are missing. Run `python train_model.py` before "
        "starting the Flask app."
    )

model = joblib.load(MODEL_PATH)
metadata = load_metadata()
currencies = metadata["currencies"]

NUMERIC_LABELS = {
    "employees": "Number of employees",
    "employees_year": "Employee data year",
    "net_income": "Net income (optional)",
    "net_income_year": "Net income year (optional)",
    "market_cap": "Market capitalization (optional)",
    "legal_units_count": "Number of legal units (optional)",
    "direct_subsidiaries_count": "Direct subsidiaries (optional)",
    "max_hierarchy_depth": "Maximum hierarchy depth (optional)",
    "revenue_year": "Year to predict",
}


def blank_form_values():
    return {feature: "" for feature in FEATURE_COLUMNS}


def parse_number(raw_value, label, required, errors, allow_negative=False, integer=False):
    value = raw_value.strip()
    if not value:
        if required:
            errors.append(f"{label} is required.")
        return None
    try:
        number = float(value)
    except ValueError:
        errors.append(f"{label} must be a number.")
        return None
    if not math.isfinite(number):
        errors.append(f"{label} must be a finite number.")
        return None
    if number < 0 and not allow_negative:
        errors.append(f"{label} cannot be negative.")
        return None
    if integer and not number.is_integer():
        errors.append(f"{label} must be a whole number.")
        return None
    return number


@app.get("/")
def home():
    return render_template("home.html", training_rows=metadata["training_rows"])


@app.get("/predict")
def predict():
    return render_template(
        "predict.html",
        currencies=currencies,
        values=blank_form_values(),
        errors=[],
        numeric_labels=NUMERIC_LABELS,
    )


@app.route("/result", methods=["GET", "POST"])
def result():
    if request.method == "GET":
        return redirect(url_for("predict"))

    values = {feature: request.form.get(feature, "").strip() for feature in FEATURE_COLUMNS}
    errors = []
    numeric_values = {}
    integer_features = {
        "employees",
        "employees_year",
        "net_income_year",
        "legal_units_count",
        "direct_subsidiaries_count",
        "revenue_year",
    }

    for feature in NUMERIC_FEATURES:
        required = feature in {"employees", "revenue_year"}
        numeric_values[feature] = parse_number(
            values[feature],
            NUMERIC_LABELS[feature],
            required,
            errors,
            allow_negative=feature == "net_income",
            integer=feature in integer_features,
        )

    for feature in CATEGORICAL_FEATURES:
        values[feature] = values[feature].strip()

    required_categories = {
        "enterprise_group_jurisdiction": "Country or jurisdiction",
        "company_type": "Company type",
        "revenue_currency": "Revenue currency",
    }
    for feature, label in required_categories.items():
        if not values[feature]:
            errors.append(f"{label} is required.")

    if values["company_type"] and values["company_type"] not in {
        "PUBLIC",
        "PRIVATE",
        "NONPROFIT",
        "STATE_OWNED",
        "OTHER",
    }:
        errors.append("Choose a valid company type.")

    if values["is_standalone_entity"] not in {"", "True", "False"}:
        errors.append("Choose Yes or No for standalone entity.")

    if values["revenue_currency"] and values["revenue_currency"] not in currencies:
        errors.append("Choose a revenue currency from the available options.")

    revenue_year = numeric_values["revenue_year"]
    if revenue_year is not None and not 1900 <= revenue_year <= 2200:
        errors.append("Year to predict must be between 1900 and 2200.")

    if errors:
        return (
            render_template(
                "predict.html",
                currencies=currencies,
                values=values,
                errors=errors,
                numeric_labels=NUMERIC_LABELS,
            ),
            400,
        )

    input_row = {
        **numeric_values,
        **{
            feature: values[feature] or None
            for feature in CATEGORICAL_FEATURES
        },
    }
    input_data = pd.DataFrame([input_row], columns=FEATURE_COLUMNS)
    for feature in CATEGORICAL_FEATURES:
        input_data[feature] = input_data[feature].fillna("").astype(str)

    prediction = max(0.0, float(model.predict(input_data)[0]))
    predicted_revenue = math.expm1(prediction)
    return render_template(
        "result.html",
        revenue=predicted_revenue,
        currency=values["revenue_currency"],
        company_type=values["company_type"],
        jurisdiction=values["enterprise_group_jurisdiction"],
        year=int(numeric_values["revenue_year"]),
    )


if __name__ == "__main__":
    app.run(debug=True)
