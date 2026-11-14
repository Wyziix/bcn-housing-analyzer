"""Market benchmarks: Airbnb (short-term) vs Idealista (long-term) rent in Barcelona.

DATA NOTE: the CSV figures are approximate, for demo. They are NOT verified
official data. Before presenting them as official, replace them with the real
Idealista Barcelona price report and the Open Data BCN export, and record the
download date and link here:
    Idealista source / date:  TODO
    Open Data BCN source / date: TODO
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
IDEALISTA_PATH = BASE_DIR / "data" / "idealista_oct2026.csv"
HISTORY_PATH = BASE_DIR / "data" / "bcn_rent_history.csv"

DAYS_PER_MONTH = 30


def load_idealista_benchmarks(path=IDEALISTA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def compare_airbnb_vs_idealista(
    district: str,
    sqm: float,
    predicted_nightly_price: float,
    occupancy_rate: float = 0.70,
    idealista_path=IDEALISTA_PATH,
) -> dict:
    """Compare predicted monthly Airbnb revenue with Idealista long-term rent.

    occupancy_rate is an assumption (default 70%), adjustable by the caller.
    """
    df = pd.read_csv(idealista_path)
    row = df[df["district"].str.lower() == district.lower()]

    if row.empty:
        raise ValueError(f"District '{district}' not found in Idealista benchmark data.")

    rent_eur_m2 = float(row["rent_eur_m2"].iloc[0])
    sale_eur_m2 = float(row["sale_eur_m2"].iloc[0])
    yoy_change = float(row["yoy_rent_change_pct"].iloc[0])

    idealista_monthly_rent = round(sqm * rent_eur_m2, 2)
    estimated_property_value = round(sqm * sale_eur_m2, 2)
    occupied_nights = DAYS_PER_MONTH * occupancy_rate
    airbnb_monthly_revenue = round(predicted_nightly_price * occupied_nights, 2)

    monthly_diff = round(airbnb_monthly_revenue - idealista_monthly_rent, 2)
    premium_pct = round(monthly_diff / idealista_monthly_rent * 100, 1)

    # Gross annual yield = annual rent / property value (before costs and taxes)
    long_term_yield = round(idealista_monthly_rent * 12 / estimated_property_value * 100, 2)
    short_term_yield = round(airbnb_monthly_revenue * 12 / estimated_property_value * 100, 2)

    return {
        "district": row["district"].iloc[0],
        "sqm": sqm,
        "idealista_rent_eur_m2": rent_eur_m2,
        "idealista_monthly_rent": idealista_monthly_rent,
        "estimated_property_value": estimated_property_value,
        "airbnb_predicted_nightly": round(predicted_nightly_price, 2),
        "occupancy_rate_pct": round(occupancy_rate * 100),
        "airbnb_monthly_revenue": airbnb_monthly_revenue,
        "monthly_difference_eur": monthly_diff,
        "airbnb_premium_pct": premium_pct,
        "yoy_rent_growth_pct": yoy_change,
        "long_term_gross_yield_pct": long_term_yield,
        "short_term_gross_yield_pct": short_term_yield,
    }


def get_district_history(district: str, history_path=HISTORY_PATH) -> pd.DataFrame:
    """Return the district's EUR/m2 trend next to the Barcelona average.

    The city average is a simple mean of the districts (not weighted).
    """
    df = pd.read_csv(history_path)

    dist_df = df[df["district"].str.lower() == district.lower()]
    if dist_df.empty:
        raise ValueError(f"District '{district}' not found in history data.")

    city_avg = (
        df.groupby("year", as_index=False)["rent_eur_m2"]
        .mean()
        .round(2)
        .rename(columns={"rent_eur_m2": "Barcelona Avg (EUR/m2)"})
    )
    dist_df = dist_df[["year", "rent_eur_m2"]].rename(
        columns={"rent_eur_m2": f"{district} (EUR/m2)"}
    )
    return pd.merge(dist_df, city_avg, on="year")


if __name__ == "__main__":
    # Quick local test: 75 m2 flat in Eixample predicted at 135 EUR/night
    result = compare_airbnb_vs_idealista("Eixample", sqm=75, predicted_nightly_price=135.0)
    print("Benchmark Comparison:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    print("\nHistorical Trend:")
    print(get_district_history("Eixample"))
