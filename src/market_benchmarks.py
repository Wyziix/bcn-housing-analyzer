"""Market benchmarks: Airbnb (short-term) vs Idealista (long-term) rent in Barcelona.

DATA SOURCES (accessed 2026-11-14)
- data/idealista_oct2026.csv  (rent and sale EUR/m2 per district, rent YoY %)
    Idealista, price reports for Barcelona, October 2026:
    https://www.idealista.com/sala-de-prensa/informes-precio-vivienda/alquiler/cataluna/barcelona-provincia/barcelona/
    https://www.idealista.com/sala-de-prensa/informes-precio-vivienda/venta/cataluna/barcelona-provincia/barcelona/
    City averages in the reports: rent 20.5 EUR/m2, sale 5,463 EUR/m2.
- data/bcn_rent_history.csv  (average monthly rent in EUR per district, 2021-2026)
    Barcelona Dades (Ajuntament de Barcelona), "Preu mitja (EUR) del lloguer
    d'habitatges". Original source: Generalitat de Catalunya, Secretaria
    d'Habitatge, from rental deposits at INCASOL. Licence CC BY 4.0. Updated 03/11/2026.
    https://portaldades.ajuntament.barcelona.cat/ca/estadistiques/b37xv8wcjh

CAVEATS
- The history is in EUR per MONTH (not EUR/m2) and 2026 is a partial year.
  The Idealista figures are asking prices in EUR/m2; the two series differ in
  source and unit, so compare trends, not levels.
- The history file includes an official "Barcelona" city row, used as the city line.
- Occupancy (default 70%) is an adjustable assumption, not a sourced figure.
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
IDEALISTA_PATH = BASE_DIR / "data" / "idealista_oct2026.csv"
HISTORY_PATH = BASE_DIR / "data" / "bcn_rent_history.csv"

DAYS_PER_MONTH = 30
CITY_ROW = "Barcelona"


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
    """Return the district's avg monthly rent (EUR/month) next to the city line.

    Columns: year, "<district> (EUR/month)", "Barcelona (EUR/month)".
    """
    df = pd.read_csv(history_path)

    dist_df = df[df["district"].str.lower() == district.lower()]
    if dist_df.empty:
        raise ValueError(f"District '{district}' not found in history data.")

    city_df = (
        df[df["district"] == CITY_ROW][["year", "avg_rent_eur_month"]]
        .rename(columns={"avg_rent_eur_month": f"{CITY_ROW} (EUR/month)"})
    )
    dist_df = dist_df[["year", "avg_rent_eur_month"]].rename(
        columns={"avg_rent_eur_month": f"{district} (EUR/month)"}
    )
    return pd.merge(dist_df, city_df, on="year")


if __name__ == "__main__":
    # Quick local test: 75 m2 flat in Eixample predicted at 135 EUR/night
    result = compare_airbnb_vs_idealista("Eixample", sqm=75, predicted_nightly_price=135.0)
    print("Benchmark Comparison:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    print("\nHistorical Trend:")
    print(get_district_history("Eixample"))
