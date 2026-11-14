# Barcelona Housing & Rental Yield Analyzer

A Streamlit app that predicts the fair nightly price of a Barcelona Airbnb listing with machine learning, tells you if a listing is a good deal or overpriced, and compares short-term (Airbnb) income with long-term rent (Idealista) for the same flat.

EADA BBAI — AI coding group project, class G1. Presentation 16 Nov 2026, final delivery 30 Nov 2026.

## Who it's for

- **Renters / visitors:** is this Airbnb listing priced fairly for its district, size and amenities?
- **Owners / investors:** for a flat in a given district, how does Airbnb income compare to a long-term lease, and what is the gross yield of each?

## Team

| Role | Member | Owns |
|---|---|---|
| #1 Data engineer (Airbnb) | David Barnett Salar | `src/clean_airbnb.py`, `data/airbnb_clean.csv` |
| #2 Market benchmarks (Idealista + history) | Igor Stachowiak | `src/market_benchmarks.py`, `data/idealista_*.csv`, `data/bcn_rent_history.csv` |
| #3 ML engineer | Krzysztof Korolec | `src/train_model.py`, `models/` |
| #4 Frontend (Streamlit) | Linus Dauder Orteu | `app.py` |
| #5 Repo lead, QA, presentation | Gleb Kopenkin | repo setup, `README.md`, `requirements.txt`, slides |

## Project structure

```
bcn-housing-analyzer/
├── data/
│   ├── raw/                    # Inside Airbnb .csv.gz downloads (gitignored, local only)
│   ├── airbnb_clean.csv        # output of #1
│   ├── idealista_<date>.csv    # #2 — district €/m² benchmarks, with source + date
│   └── bcn_rent_history.csv    # #2 — yearly rent €/m² by district (Open Data BCN)
├── models/
│   ├── price_model.pkl         # output of #3 (sklearn Pipeline)
│   └── feature_importance.csv  # output of #3
├── src/
│   ├── clean_airbnb.py         # #1
│   ├── market_benchmarks.py    # #2
│   └── train_model.py          # #3
├── app.py                      # #4
└── requirements.txt
```

## How to run

```bash
git clone https://github.com/Wyziix/bcn-housing-analyzer.git
cd bcn-housing-analyzer
pip install -r requirements.txt

# 1. Put Inside Airbnb Barcelona listings.csv.gz files in data/raw/
python src/clean_airbnb.py      # -> data/airbnb_clean.csv
python src/train_model.py       # -> models/price_model.pkl
streamlit run app.py
```

Always run commands from the repo root.

## Data sources

- **Inside Airbnb** — Barcelona `listings.csv.gz` snapshots: http://insideairbnb.com/get-the-data
- **Idealista** — Barcelona rent and sale price reports by district (€/m²), October 2026. Links in `src/market_benchmarks.py`; screenshot of the rent table in `docs/idealista_oct2026_source.png`.
- **Barcelona Dades (Ajuntament)** — average monthly rent in € by district, 2021–2026 (INCASOL rental deposits, CC BY 4.0): https://portaldades.ajuntament.barcelona.cat/ca/estadistiques/b37xv8wcjh

## Model results

_To be filled in by #3 after training (test set, latest snapshot only)._

| Model | MAE (€/night) | RMSE (€/night) | R² |
|---|---|---|---|
| Ridge (baseline) | | | |
| Random Forest | | | |

## Limitations

- Airbnb data has no m² column, so the ML price uses guests / bedrooms / bathrooms. The m² input only affects the Idealista long-term comparison.
- Occupancy (default 70%) is a user assumption, not measured data.
- Airbnb income is gross: no platform fees, cleaning, taxes or vacancy costs.
- Barcelona will not renew its ~10,000 tourist-flat licences when they expire in November 2028, so the short-term vs long-term comparison is about today's market, not a long-term investment case.
- Barcelona is a regulated (rent-capped) zone, so long-term asking rents may differ from what new contracts can legally charge.

## How we used AI and GitHub

- Each member generated their module with AI coding tools, then reviewed and tested it before opening a pull request.
- One owner per file, one branch per person, merges into `main` through pull requests (see `CONTRIBUTING.md`).
- AI mistakes we found and fixed: see `CONTRIBUTING.md` → "Known AI mistakes".
