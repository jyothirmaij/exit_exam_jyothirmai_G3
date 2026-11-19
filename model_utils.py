from pathlib import Path

import json


ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "revenue_model.joblib"
METADATA_PATH = MODEL_DIR / "revenue_model_metadata.json"

NUMERIC_FEATURES = [
    "employees",
    "employees_year",
    "net_income",
    "net_income_year",
    "market_cap",
    "legal_units_count",
    "direct_subsidiaries_count",
    "max_hierarchy_depth",
    "revenue_year",
]

CATEGORICAL_FEATURES = [
    "enterprise_group_jurisdiction",
    "company_type",
    "net_income_currency",
    "market_cap_currency",
    "revenue_currency",
    "is_standalone_entity",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_metadata():
    with METADATA_PATH.open(encoding="utf-8") as metadata_file:
        return json.load(metadata_file)
