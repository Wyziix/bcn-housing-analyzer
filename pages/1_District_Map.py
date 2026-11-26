"""Streamlit page for exploring Barcelona's Airbnb listing prices."""

import altair as alt
import pydeck as pdk
import streamlit as st

from src.district_stats import (
    ENTIRE_HOME,
    PRICE_COLOUR_MAX,
    PRICE_COLOUR_MIN,
    add_price_colour,
    district_summary,
    filter_listings,
    load_listings,
)


st.set_page_config(page_title="District Map", layout="wide")


@st.cache_data
def load_map_listings():
    return load_listings()


st.title("District Map")
listings = load_map_listings()
district_options = sorted(listings["district"].dropna().unique())
room_options = sorted(listings["room_type"].dropna().unique())

with st.sidebar:
    st.header("Listing filters")
    districts = st.multiselect("Districts", district_options, default=district_options)
    room_types = st.multiselect(
        "Listing type",
        room_options,
        default=[ENTIRE_HOME] if ENTIRE_HOME in room_options else room_options,
    )
    price_range = st.slider("Nightly price (€)", min_value=20, max_value=600, value=(20, 600))

filtered = filter_listings(listings, districts, room_types, price_range)
if filtered.empty:
    st.warning("No listings match these filters. Try a wider price range or change your selections.")
    st.stop()

summary = district_summary(filtered)
count_metric, median_metric, district_metric = st.columns([1, 1, 2])
count_metric.metric("Listings shown", f"{len(filtered):,}")
median_metric.metric("Median nightly price", f"€{filtered['price'].median():,.2f}")
district_metric.metric("Most expensive district", summary.iloc[0]["district"])

map_data = add_price_colour(filtered)[
    ["longitude", "latitude", "district", "room_type", "price", "price_colour"]
].copy()
map_data["price_label"] = map_data.pop("price").map("€{:,.2f} / night".format)
scatterplot = pdk.Layer(
    "ScatterplotLayer",
    map_data,
    get_position="[longitude, latitude]",
    get_fill_color="price_colour",
    get_radius=35,
    radius_min_pixels=3,
    radius_max_pixels=8,
    opacity=0.7,
    pickable=True,
)
st.pydeck_chart(
    pdk.Deck(
        layers=[scatterplot],
        initial_view_state=pdk.ViewState(latitude=41.395, longitude=2.165, zoom=11.5),
        map_style="light",
        tooltip={"text": "{district}\n{room_type}\n{price_label}"},
    )
)
st.caption(f"Blue: €{PRICE_COLOUR_MIN} and below · Red: €{PRICE_COLOUR_MAX} and above")

st.subheader("Median Airbnb price by district")
chart = alt.Chart(summary).mark_bar().encode(
    x=alt.X("median_price:Q", title="Median Airbnb price (€/night)"),
    y=alt.Y("district:N", title=None, sort=summary["district"].tolist()),
    tooltip=[
        alt.Tooltip("district:N", title="District"),
        alt.Tooltip("median_price:Q", title="€/night", format=",.2f"),
    ],
)
st.altair_chart(chart)

st.subheader("Airbnb and Idealista rents by district")
st.caption(
    "Airbnb medians reflect the current filters. "
    "Idealista shows October 2026 long-term asking rent per m² per month."
)
st.dataframe(
    summary,
    hide_index=True,
    column_config={
        "district": "District",
        "listings_count": st.column_config.NumberColumn("Listings", format="%d"),
        "median_price": st.column_config.NumberColumn("Airbnb median (€/night)", format="%.2f"),
        "median_price_per_guest": st.column_config.NumberColumn("Median per guest (€/night)", format="%.2f"),
        "entire_homes_pct": st.column_config.NumberColumn("Entire homes (%)", format="%.1f"),
        "rent_eur_m2": st.column_config.NumberColumn("Idealista rent (€/m²/month)", format="%.2f"),
    },
)

with st.expander("Explore neighbourhoods"):
    district = st.selectbox("District to explore", summary["district"].tolist())
    neighbourhoods = (
        filtered.loc[filtered["district"].eq(district)]
        .groupby("neighborhood", as_index=False)
        .agg(listings_count=("price", "size"), median_price=("price", "median"))
        .sort_values(["median_price", "neighborhood"], ascending=[False, True])
    )
    st.dataframe(
        neighbourhoods,
        hide_index=True,
        column_config={
            "neighborhood": "Neighbourhood",
            "listings_count": st.column_config.NumberColumn("Listings", format="%d"),
            "median_price": st.column_config.NumberColumn("Airbnb median (€/night)", format="%.2f"),
        },
    )
