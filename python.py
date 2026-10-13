import pandas as pd
import numpy as np

print("Capstone project")

# 1. Load Dataset
df = pd.read_csv('RICE_merged.csv')

# --- DATASET SUMMARY (BEFORE CLEANING) ---
print("\n--- DATASET SUMMARY (BEFORE CLEANING) ---")
print("Dataset Shape:", df.shape)
print("Duplicate Rows Count:", df.duplicated().sum())
print("Duplicate Columns Count:", df.columns.duplicated().sum())
print("Total Missing Values:", df.isnull().sum().sum())
print("Missing Values Per Column:")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\n--- FIRST 5 ROWS (BEFORE CLEANING) ---")
print(df.head())


# 2. Level 4 Efficient Vectorized Data Cleaning

# Step A: Remove Duplicates
df = df.drop_duplicates()

# Step B: Outlier Correction - Sunshine Hours (> 24 hrs)
# Vectorized replacement using np.where and group mean transform
df['SSH(hrs)'] = np.where(df['SSH(hrs)'] > 24, np.nan, df['SSH(hrs)'])
df['SSH(hrs)'] = df['SSH(hrs)'].fillna(
    df.groupby(['Location', 'Standard Week'])['SSH(hrs)'].transform('mean')
)

# Step C: Outlier Correction - Extreme Max Temperature (> 50°C)
# Vectorized replacement using np.where
df['MaxT'] = np.where(df['MaxT'] > 50, df['NASA_MaxT'], df['MaxT'])

# Step D: Missing Value Imputation - NASA Weather Metrics
# Vectorized multi-column group mean transformation
nasa_cols = ['NASA_MaxT', 'NASA_MinT', 'NASA_RH', 'NASA_RF_mm', 'NASA_WS_kmph']
group_means = df.groupby(['Location', 'Standard Week'])[nasa_cols].transform('mean')

df[nasa_cols] = df[nasa_cols].fillna(group_means)
df[nasa_cols] = df[nasa_cols].fillna(df[nasa_cols].mean())


# 3. Save Cleaned Dataset
df.to_csv('RICE_merged_cleaned.csv', index=False)


# --- DATASET SUMMARY (AFTER CLEANING) ---
print("\n--- DATASET SUMMARY (AFTER CLEANING) ---")
print("Cleaned Dataset Shape:", df.shape)
print("Duplicate Rows Remaining:", df.duplicated().sum())
print("Total Missing Values Remaining:", df.isnull().sum().sum())

print("\n--- FIRST 5 ROWS (AFTER CLEANING) ---")
print(df.head())

print("\nDataset cleaned and saved successfully as 'RICE_merged_cleaned.csv'!")