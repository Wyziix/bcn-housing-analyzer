"""
src/train_model.py  (Teammate #3 - ML Engineer)

Trains Ridge (baseline) vs Random Forest vs XGBoost on data/airbnb_clean.csv,
saves the WINNING pipeline to models/price_model.pkl and feature
importances to models/feature_importance.csv.

Run from the project root:
    python src/train_model.py
"""

import os

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

DATA_PATH = "data/airbnb_clean.csv"
MODEL_OUTPUT_PATH = "models/price_model.pkl"
IMPORTANCE_OUTPUT_PATH = "models/feature_importance.csv"
METRICS_OUTPUT_PATH = "models/metrics.csv"

# Exact 12 features sent by Teammate #4's app.py sidebar
CAT_FEATURES = ["district", "room_type"]
NUM_FEATURES = [
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
    "review_scores_rating",
    "number_of_reviews",
    "has_ac",
    "has_wifi",
    "has_elevator",
    "has_kitchen",
]
TARGET = "price"


def build_preprocessor() -> ColumnTransformer:
    """Fresh preprocessor per pipeline so the models never share state."""
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
            ("num", StandardScaler(), NUM_FEATURES),
        ]
    )


def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run Teammate #1's cleaning script first."
        )

    df = pd.read_csv(DATA_PATH)

    missing = [c for c in CAT_FEATURES + NUM_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Columns missing from {DATA_PATH}: {missing}")

    # Make sure numeric columns (incl. has_* flags) are really numeric
    for col in NUM_FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=CAT_FEATURES + NUM_FEATURES + [TARGET])
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with missing/invalid values.")

    print(f"Loaded {len(df)} listings.")
    return df


def evaluate_model(name: str, model: Pipeline, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"\n--- {name} ---")
    print(f"MAE  (Avg Error): EUR {mae:.2f} / night")
    print(f"RMSE (Penalized): EUR {rmse:.2f} / night")
    print(f"R2 Score:         {r2:.3f}")
    return {"model": name, "mae": mae, "rmse": rmse, "r2": r2}


def get_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """Works for tree models (RF, XGBoost: feature_importances_) and Ridge (|coefficients|)."""
    pre = pipeline.named_steps["preprocessor"]
    reg = pipeline.named_steps["regressor"]
    names = list(pre.named_transformers_["cat"].get_feature_names_out(CAT_FEATURES))
    names += NUM_FEATURES

    if hasattr(reg, "feature_importances_"):
        values = reg.feature_importances_
    else:
        values = np.abs(reg.coef_)
        values = values / values.sum()  # normalise so it reads like importance

    return (
        pd.DataFrame({"feature": names, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def main():
    print(f"scikit-learn version: {sklearn.__version__}")
    df = load_data()

    # Keep only the latest snapshot: the same flat appears in every snapshot,
    # so mixing them would put the same flat in both train and test (leakage).
    if "snapshot" in df.columns:
        latest = sorted(df["snapshot"].unique())[-1]
        df = df[df["snapshot"] == latest]
        print(f"Using latest snapshot only: {latest} ({len(df)} listings)")
    else:
        print("WARNING: no 'snapshot' column found, using all rows.")

    X = df[CAT_FEATURES + NUM_FEATURES]
    y = df[TARGET]

    # 80% train / 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    pipelines = {
        "Ridge Regression (Baseline)": Pipeline(
            [("preprocessor", build_preprocessor()), ("regressor", Ridge(alpha=1.0))]
        ),
        "Random Forest Regressor": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=150,
                        max_depth=16,
                        min_samples_leaf=5,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "XGBoost Regressor": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "regressor",
                    XGBRegressor(
                        n_estimators=300,
                        max_depth=6,
                        learning_rate=0.05,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    results = []
    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        results.append(evaluate_model(name, pipe, X_test, y_test))

    # Pick the winner by lowest test MAE
    best = min(results, key=lambda r: r["mae"])
    best_pipeline = pipelines[best["model"]]
    print(f"\nWINNER: {best['model']} (lowest MAE: EUR {best['mae']:.2f})")

    # Save model + metrics
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_pipeline, MODEL_OUTPUT_PATH, compress=3)
    size_mb = os.path.getsize(MODEL_OUTPUT_PATH) / (1024 * 1024)
    print(f"Saved pipeline to {MODEL_OUTPUT_PATH} ({size_mb:.1f} MB)")
    if size_mb > 90:
        print("WARNING: file is close to GitHub's 100 MB limit. Reduce max_depth/n_estimators.")

    pd.DataFrame(results).to_csv(METRICS_OUTPUT_PATH, index=False)

    # Feature importances
    fi_df = get_feature_importance(best_pipeline)
    fi_df.to_csv(IMPORTANCE_OUTPUT_PATH, index=False)
    print("\nTop 8 Price Drivers:")
    print(fi_df.head(8).to_string(index=False))

    print("\nNumbers for the slides / Jordi's Q&A:")
    print(f"  Model: {best['model']}")
    print(f"  MAE:   EUR {best['mae']:.2f}")
    print(f"  RMSE:  EUR {best['rmse']:.2f}")
    print(f"  R2:    {best['r2']:.3f}")


if __name__ == "__main__":
    main()
