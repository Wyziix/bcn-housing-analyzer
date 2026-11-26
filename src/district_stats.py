"""Filter Barcelona Airbnb listings and summarize prices by district."""

from collections.abc import Collection
from pathlib import Path

import pandas as pd

if __package__:
    from .market_benchmarks import load_idealista_benchmarks
else:
    from market_benchmarks import load_idealista_benchmarks


LISTINGS_PATH = Path(__file__).resolve().parents[1] / "data" / "airbnb_clean.csv"
ENTIRE_HOME = "Entire home/apt"
PRICE_COLOUR_MIN = 50
PRICE_COLOUR_MAX = 350


def load_listings() -> pd.DataFrame:
    """Load mappable listings from the latest available snapshot."""
    listings = pd.read_csv(LISTINGS_PATH)
    snapshots = sorted(listings["snapshot"].dropna().unique())
    if not snapshots:
        return listings.iloc[:0]

    latest = listings.loc[listings["snapshot"].eq(snapshots[-1])]
    return latest.dropna(subset=["latitude", "longitude", "price"])


def filter_listings(
    df: pd.DataFrame,
    districts: Collection[str],
    room_types: Collection[str],
    price_range: tuple[int | float, int | float],
) -> pd.DataFrame:
    """Keep selected districts and room types within the inclusive price range."""
    min_price, max_price = price_range
    return df.loc[
        df["district"].isin(districts)
        & df["room_type"].isin(room_types)
        & df["price"].between(min_price, max_price)
    ].copy()


def add_price_colour(df: pd.DataFrame) -> pd.DataFrame:
    """Add an RGB price colour, using blue at €50 and red at €350."""
    clipped_price = df["price"].clip(PRICE_COLOUR_MIN, PRICE_COLOUR_MAX)
    fraction = (clipped_price - PRICE_COLOUR_MIN) / (PRICE_COLOUR_MAX - PRICE_COLOUR_MIN)
    red = (fraction * 255).round().astype(int)
    return df.assign(price_colour=[[value, 0, 255 - value] for value in red])


def district_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Rank districts by median nightly price and attach Idealista asking rents."""
    guests = df["accommodates"].where(df["accommodates"].gt(0))
    listings = df.assign(
        price_per_guest=df["price"].div(guests),
        entire_home=df["room_type"].eq(ENTIRE_HOME),
    )
    summary = listings.groupby("district", as_index=False, observed=True).agg(
        listings_count=("price", "size"),
        median_price=("price", "median"),
        median_price_per_guest=("price_per_guest", "median"),
        entire_homes_pct=("entire_home", "mean"),
    )
    summary["entire_homes_pct"] *= 100
    rents = load_idealista_benchmarks()[["district", "rent_eur_m2"]]
    return summary.merge(rents, on="district", how="left", validate="one_to_one").sort_values(
        ["median_price", "district"], ascending=[False, True], ignore_index=True
    )


if __name__ == "__main__":
    print(district_summary(load_listings()).to_string(index=False, float_format="{:.2f}".format))
