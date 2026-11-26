

import os
import joblib
import pandas as pd
import streamlit as st
from src.market_benchmarks import compare_airbnb_vs_idealista, get_district_history


st.set_page_config(
    page_title="Barcelona Housing & Rental Yield Analyzer",
    page_icon="🏠",
    layout="wide",
)


DISTRICTS = [
    "Ciutat Vella",
    "Eixample",
    "Sant Martí",
    "Sarrià-Sant Gervasi",
    "Les Corts",
    "Gràcia",
    "Sants-Montjuïc",
    "Horta-Guinardó",
    "Sant Andreu",
    "Nou Barris",
]


MODEL_PATH = "models/price_model.pkl"


@st.cache_resource
def load_model(path: str):
    if os.path.exists(path):
        return joblib.load(path)
    return None


def predict_nightly_rate(model, features: dict) -> float:
    """Uses trained ML pipeline if available; otherwise uses a baseline heuristic for UI testing."""
    if model is not None:
        input_df = pd.DataFrame([features])
        return float(model.predict(input_df)[0])

    # Temporary fallback before Teammate #3 pushes price_model.pkl
    base = 65.0 + (features["bedrooms"] * 28.0) + (features["accommodates"] * 8.0)
    if features["room_type"] == "Entire home/apt":
        base *= 1.35
    if features["district"] in ["Eixample", "Ciutat Vella", "Sarrià-Sant Gervasi"]:
        base *= 1.15
    base += features["has_ac"] * 10 + features["has_elevator"] * 6
    return round(base, 2)


# --- HEADER ---
st.title("🏠 Barcelona Housing & Rental Yield Analyzer")
st.caption("ML Fair-Price Predictor vs. Actual Idealista October 2026 Market Benchmarks")


model = load_model(MODEL_PATH)
if model is None:
    st.info("Running in UI preview mode (fallback estimator). Once `models/price_model.pkl` is added, live ML predictions activate automatically.")


# --- SIDEBAR INPUTS ---
st.sidebar.header("1. Apartment Specifications")
district = st.sidebar.selectbox("District", DISTRICTS, index=1)
room_type = st.sidebar.selectbox("Listing Type", ["Entire home/apt", "Private room", "Shared room"])
sqm = st.sidebar.slider("Size (m²)", min_value=25, max_value=200, value=75, step=5)
st.sidebar.caption("Only affects the Idealista comparison, not the predicted nightly price.")  # FIX 4
accommodates = st.sidebar.slider("Guests Accommodated", min_value=1, max_value=10, value=4)
bedrooms = st.sidebar.slider("Bedrooms", min_value=0, max_value=6, value=2)
bathrooms = st.sidebar.slider("Bathrooms", min_value=1.0, max_value=4.0, value=1.0, step=0.5)
number_of_reviews = st.sidebar.slider("Number of reviews", min_value=0, max_value=500, value=25)
review_scores_rating = st.sidebar.slider("Review score", min_value=1.0, max_value=5.0, value=4.8, step=0.1)


st.sidebar.header("2. Amenities")
has_ac = int(st.sidebar.checkbox("Air Conditioning", value=True))
has_wifi = int(st.sidebar.checkbox("High-Speed Wi-Fi", value=True))
has_elevator = int(st.sidebar.checkbox("Elevator", value=True))
has_kitchen = int(st.sidebar.checkbox("Full Kitchen", value=True))


st.sidebar.header("3. Deal Checker & Occupancy")
actual_listing_price = st.sidebar.number_input("Actual Airbnb Listing Price (€/night)", min_value=20, max_value=600, value=145)
occupancy_pct = st.sidebar.slider("Expected Monthly Occupancy (%)", min_value=40, max_value=95, value=70, step=5)


# --- PREDICTION & BENCHMARKING ---
features = {
    "district": district,
    "room_type": room_type,
    "accommodates": accommodates,
    "bedrooms": bedrooms,
    "beds": max(1, bedrooms),
    "bathrooms": bathrooms,
    "review_scores_rating": review_scores_rating,
    "number_of_reviews": number_of_reviews,
    "has_ac": has_ac,
    "has_wifi": has_wifi,
    "has_elevator": has_elevator,
    "has_kitchen": has_kitchen,
}


predicted_nightly = predict_nightly_rate(model, features)
price_gap = round(actual_listing_price - predicted_nightly, 2)


benchmark = compare_airbnb_vs_idealista(
    district=district,
    sqm=sqm,
    predicted_nightly_price=predicted_nightly,
    occupancy_rate=occupancy_pct / 100.0,
)


# --- TOP KPI CARDS ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Predicted Fair Nightly Price", f"€{predicted_nightly:.2f}", f"{-price_gap:+.2f} € vs listing")
col2.metric(f"Est. Airbnb Monthly ({occupancy_pct}% occ.)", f"€{benchmark['airbnb_monthly_revenue']:,.0f}")
col3.metric("Idealista Long-Term Rent (Oct 2026)", f"€{benchmark['idealista_monthly_rent']:,.0f}", f"{benchmark['idealista_rent_eur_m2']} €/m²")
col4.metric("Short-Term vs. Long-Term Gap", f"{benchmark['monthly_difference_eur']:+,.0f} €/mo", f"{benchmark['airbnb_premium_pct']:+.1f}%")


# --- DEAL VERDICT BANNER ---
if price_gap <= -10:
    st.success(f"**Good Deal:** At €{actual_listing_price}/night, this listing is **€{abs(price_gap):.2f}/night below** the predicted fair value for {district}.")
elif price_gap >= 10:
    st.warning(f"**Overpriced:** At €{actual_listing_price}/night, this listing is **€{price_gap:.2f}/night above** the predicted fair value for {district}.")
else:
    st.info(f"**Fair Market Price:** At €{actual_listing_price}/night, this listing aligns closely with predicted market rates in {district}.")


st.divider()


# --- CHARTS & YIELD COMPARISON ---
left_col, right_col = st.columns([1.3, 1])


with left_col:
    st.subheader(f"Multi-Year Rent Evolution (€/month): {district} vs. Barcelona city")  # FIX 3
    history_df = get_district_history(district).set_index("year")
    st.line_chart(history_df)


with right_col:
    st.subheader("Investment Yield Breakdown (Idealista Oct 2026)")
    summary_table = pd.DataFrame({
        "Strategy": ["Short-Term (Airbnb ML)", "Long-Term (Idealista Lease)"],
        "Monthly Revenue (€)": [benchmark["airbnb_monthly_revenue"], benchmark["idealista_monthly_rent"]],
        "Gross Annual Yield (%)": [f"{benchmark['short_term_gross_yield_pct']}%", f"{benchmark['long_term_gross_yield_pct']}%"],
    })
    st.dataframe(summary_table, hide_index=True)  # FIX 2: removed deprecated use_container_width
    st.caption(f"YoY district rent growth in {district}: {benchmark['yoy_rent_growth_pct']:+}% (Idealista Oct 2026).")  # FIX 1
