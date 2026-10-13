# Agricultural Pest Outbreak Analysis and Forecasting Using Environmental Factors

## Problem statement
Analyse rice pest outbreaks across six research stations in India and study how environmental
factors (temperature, humidity, rainfall, wind) relate to pest counts, with the aim of
supporting forecasting.

## Datasets
| Dataset | Source | Description |
|---|---|---|
| `RICE.csv` | [Kaggle: Rice Pest and Diseases](https://www.kaggle.com/datasets/zsinghrahulk/rice-pest-and-diseases/data) | Weekly pest observations with station weather, 6 locations, 11 pests |
| NASA POWER daily weather | [NASA POWER](https://power.larc.nasa.gov/) (Agroclimatology community) | T2M_MAX, T2M_MIN, RH2M, PRECTOTCORR, WS2M, 1981-2011 |

**Why combine them?** The NASA data adds an independent environmental source for the same
locations, lets the station readings be cross-checked (outlier detection), and introduces real
data-quality issues (coverage gaps before 1981, unit differences, different spatial scale).

## Repository structure
```
data/raw/RICE.csv                  original Kaggle data (unchanged)
data/raw/nasa_daily/*.csv          NASA POWER daily downloads, one per location
data/processed/                    weekly NASA data and the merged dataset
src/merge_data.py                  merging pipeline
```

## How the merge works
1. Strip NASA header text, convert `-999` to missing.
2. Convert day-of-year to Standard Week (days 1-7 = week 1, ...).
3. Aggregate daily to weekly (mean for temperature/humidity/wind, sum for rainfall; wind m/s to km/h).
4. Left-join onto RICE on Location + Observation Year + Standard Week.

Run: `pip install -r requirements.txt` then `python src/merge_data.py`

## Known notes
- NASA POWER starts in 1981, so older rows (Cuttack 1959-1975, Rajendranagar 1975-1980) have no NASA values.
- NASA values are grid averages (~50 km), so they differ somewhat from station readings.

## Next steps
Data cleaning (missing values, duplicates, outliers), feature engineering, analysis.
