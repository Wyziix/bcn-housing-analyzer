"""
src/clean_airbnb.py  (Teammate #1 - Data)

Reads Inside Airbnb Barcelona snapshots from data/raw/ and writes one clean
table to data/airbnb_clean.csv for train_model.py (Krzysztof) and app.py (Linus).

Run from the repo root:
    python src/clean_airbnb.py

Raw files go in data/raw/ (gitignored), named by date, e.g.
    data/raw/listings_2026_09.csv.gz
"""

import glob
import os
import re

import pandas as pd

RAW_GLOB = "data/raw/listings_*.csv*"
OUTPUT_PATH = "data/airbnb_clean.csv"

MIN_PRICE = 20   # below this: test / broken listings
MAX_PRICE = 600  # above this: luxury outliers that distort MAE

# Must match Igor's data/idealista_oct2026.csv exactly
DISTRICTS = {
    "Ciutat Vella", "Eixample", "Sant Martí", "Sarrià-Sant Gervasi", "Les Corts",
    "Gràcia", "Sants-Montjuïc", "Horta-Guinardó", "Sant Andreu", "Nou Barris",
}

KEEP_COLS = [
    "id", "neighbourhood_group_cleansed", "neighbourhood_cleansed", "room_type",
    "accommodates", "bedrooms", "beds", "bathrooms", "bathrooms_text",
    "amenities", "review_scores_rating", "number_of_reviews",
    "latitude", "longitude", "price",
]

# Final columns. The 12 features + price are what train_model.py checks for.
OUTPUT_COLS = [
    "id", "snapshot", "district", "neighborhood", "room_type",
    "accommodates", "bedrooms", "beds", "bathrooms",
    "review_scores_rating", "number_of_reviews",
    "has_ac", "has_wifi", "has_elevator", "has_kitchen",
    "latitude", "longitude", "price",
]


def bathrooms_from_text(text) -> float:
    """'1.5 baths' -> 1.5, 'Half-bath' -> 0.5, missing -> NaN."""
    if pd.isna(text):
        return float("nan")
    text = str(text).lower()
    if "half" in text:
        return 0.5
    match = re.search(r"\d+(\.\d+)?", text)
    return float(match.group()) if match else float("nan")


def clean_snapshot(path: str) -> pd.DataFrame:
    name = os.path.basename(path).split(".")[0]  # listings_2026_09
    df = pd.read_csv(path, usecols=lambda c: c in KEEP_COLS, low_memory=False)
    print(f"{name}: {len(df):,} raw rows")
    df["snapshot"] = name

    # Price: "$1,250.00" -> 1250.0 (Inside Airbnb shows $, values are EUR)
    df["price"] = pd.to_numeric(
        df["price"].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce"
    )
    df = df[df["price"].between(MIN_PRICE, MAX_PRICE)].copy()

    # Room types: keep the 3 the app offers (drops "Hotel room")
    df = df[df["room_type"].isin(["Entire home/apt", "Private room", "Shared room"])]

    df = df.rename(columns={
        "neighbourhood_group_cleansed": "district",
        "neighbourhood_cleansed": "neighborhood",
    })
    df = df[df["district"].isin(DISTRICTS)]

    # Bathrooms: numeric column is often empty, fall back to the text one
    if "bathrooms_text" in df.columns:
        from_text = df["bathrooms_text"].apply(bathrooms_from_text)
        df["bathrooms"] = df["bathrooms"].fillna(from_text) if "bathrooms" in df else from_text

    # Missing numbers -> median (dropping rows would lose a lot of listings)
    for col in ["bedrooms", "beds", "bathrooms", "review_scores_rating"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())
    df["number_of_reviews"] = df["number_of_reviews"].fillna(0)

    # Amenities text -> 0/1 columns
    amenities = df["amenities"].fillna("").str.lower()
    df["has_ac"] = amenities.str.contains("air conditioning").astype(int)
    df["has_wifi"] = amenities.str.contains("wifi").astype(int)
    df["has_elevator"] = amenities.str.contains("elevator").astype(int)
    df["has_kitchen"] = amenities.str.contains("kitchen").astype(int)

    print(f"{name}: {len(df):,} clean rows")
    return df[OUTPUT_COLS]


def main():
    files = sorted(glob.glob(RAW_GLOB))
    if not files:
        raise FileNotFoundError(
            "No files in data/raw/. Download listings.csv.gz for Barcelona from "
            "insideairbnb.com/get-the-data and save it as data/raw/listings_YYYY_MM.csv.gz"
        )

    full = pd.concat([clean_snapshot(f) for f in files], ignore_index=True)
    full.to_csv(OUTPUT_PATH, index=False)

    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"\nSaved {len(full):,} rows to {OUTPUT_PATH} ({size_mb:.1f} MB)")
    print("Rows per snapshot:\n" + full["snapshot"].value_counts().sort_index().to_string())
    print("Rows per district:\n" + full["district"].value_counts().to_string())
    print(f"Median price: EUR {full['price'].median():.0f} / night")


if __name__ == "__main__":
    main()
