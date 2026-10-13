Agricultural Pest Outbreak Analysis and Forecasting Using Environmental Factors
Problem statement
Analyse rice pest outbreaks across six research stations in India and study how environmental
factors (temperature, humidity, rainfall, wind) relate to pest counts, with the aim of
supporting forecasting.
Datasets
Dataset	Source	Description
`RICE.csv`	Kaggle: Rice Pest and Diseases	Weekly pest observations with station weather, 6 locations, 11 pests
NASA POWER weather data	NASA POWER	Daily temperature, humidity, rainfall and wind for the same 6 locations, 1981-2011
The NASA data was combined with the rice data to add an independent weather source for the
same locations and to cross-check the station readings.
Repository structure
```
data/raw/            original datasets (RICE.csv and NASA downloads)
data/processed/      weekly NASA data and the merged dataset
src/merge_data.py    code that merges the two datasets
```
