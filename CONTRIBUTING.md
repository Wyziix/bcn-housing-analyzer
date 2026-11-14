# How we work

## Rules

1. **Never push to `main`.** Make your own branch and open a pull request:
   - #1 `feat/data-cleaning`
   - #2 `feat/benchmarks`
   - #3 `feat/ml-model`
   - #4 `feat/streamlit-ui`
2. **Only edit the files you own** (table in README). Need a change in someone else's file? Tell them.
3. **Run everything from the repo root.** Use relative paths like `data/airbnb_clean.csv`. No `C:/Users/...`.
4. **Don't commit big files.** Raw `.csv.gz` stays in `data/raw/` (gitignored). GitHub rejects files over 100 MB.
5. **Understand every line you push.** Jordi will ask each of us about our own code.

```bash
git clone https://github.com/Wyziix/bcn-housing-analyzer.git
cd bcn-housing-analyzer
git checkout -b feat/your-branch
# ... work ...
git add <your files>
git commit -m "Short description"
git push -u origin feat/your-branch
# then open a Pull Request on GitHub
```

## Deadlines

| When | What |
|---|---|
| Sat 14 Nov, 15:00 | Push whatever you have, even if it's broken |
| Sat 14 Nov, 20:00 | Your file runs, PR open |
| Sun 15 Nov, 12:00 | Call: everyone explains their code to the group |
| Sun 15 Nov, 19:00 | Full rehearsal with live demo |
| Mon 16 Nov, 09:00 | Presentation, room A73 |
| Mon 30 Nov | Final version + README |

## Data contract (do not rename these)

**#1 → `data/airbnb_clean.csv`** must contain at least:

`district, room_type, accommodates, bedrooms, beds, bathrooms, review_scores_rating, number_of_reviews, has_ac, has_wifi, has_elevator, has_kitchen, price, snapshot, id`

- `district` = Inside Airbnb `neighbourhood_group_cleansed` (the 10 official districts, e.g. `Eixample`, `Sant Martí`).
- `snapshot` = file name like `listings_2026_09` so #3 can pick the latest one.
- `price` = float in €/night.

**#2 → `src/market_benchmarks.py`** must expose:

```python
compare_airbnb_vs_idealista(district, sqm, predicted_nightly_price, occupancy_rate=0.70) -> dict
get_district_history(district) -> pd.DataFrame
```

District names must match #1's spelling exactly.

**#3 → `models/price_model.pkl`** = a scikit-learn `Pipeline` that takes a DataFrame with the 12 feature columns above (no `price`) and returns €/night.

**#4 → `app.py`** imports #2's functions and loads #3's pickle with `joblib.load`.

## Known AI mistakes (fix these in your file)

The starter code came from an AI chat. We checked it and found these problems. Fixing them is part of the "how we reviewed AI code" slide.

**#1 clean_airbnb.py**
- Inside Airbnb shows prices with a `$` sign, but Barcelona values are in **euros**. Strip the symbol, keep the number.
- `bathrooms` is sometimes empty in newer files; there is also `bathrooms_text` (e.g. "1.5 baths"). Fill from it if you can, median otherwise.
- Keep the `id` and `snapshot` columns (#3 needs them).

**#2 market_benchmarks.py / data**
- ✅ **Fixed (PR #1).** The AI's data was a mix of real and wrong. We checked each column against the source:
  - **Rent €/m² per district: real.** Matches the Idealista October 2026 report exactly (screenshot: `docs/idealista_oct2026_source.png`).
  - **YoY rent change: wrong.** The AI had +5–9% everywhere; Idealista shows −3.0% to +7.2% (e.g. Ciutat Vella −2.2%, Eixample +0.5%). Replaced with Idealista's annual variation.
  - **Sale €/m²: wrong.** Replaced with Idealista's October 2026 sale report.
  - **2021–2026 rent history: invented.** Replaced with Barcelona Dades (Ajuntament), average monthly rent in € per district (INCASOL deposits). Note: units are €/month, not €/m².
- The chat claimed Eixample yields "~4.7% long-term vs ~6.2% short-term". With the real data the code gives **3.83% vs 6.80%** for 75 m² at €135/night and 70% occupancy. Always quote what the code prints.

**#3 train_model.py**
- **Data leakage:** the clean file has several snapshots, so the same flat appears in train and test, which inflates R². Train/test only on the **latest snapshot**:
  ```python
  latest = sorted(df["snapshot"].unique())[-1]
  df = df[df["snapshot"] == latest]
  ```
  (Older snapshots are for the trend chart only.)
- **File size:** a 150-tree, depth-16 forest can be too big for GitHub. Save with compression and check the size:
  ```python
  joblib.dump(rf_pipeline, MODEL_OUTPUT_PATH, compress=3)
  ```
  If still over ~50 MB, use `n_estimators=100, max_depth=12`.
- Write the MAE / RMSE / R² you get into the README table.

**#4 app.py**
- The m² slider doesn't change the predicted Airbnb price (no m² in Airbnb data). Add a small caption saying it only affects the Idealista comparison.
- The fallback estimator is only for testing the UI. Make sure the demo runs with the real `price_model.pkl`.
