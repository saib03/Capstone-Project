Agricultural Pest Outbreak Analysis and Forecasting Using Environmental Factors

Problem statement
Analyse rice pest outbreaks across six research stations in India and study how environmental
factors (temperature, humidity, rainfall, wind) relate to pest counts, with the aim of
supporting forecasting.

Datasets
Dataset	Source	Description
`RICE.csv`	
Kaggle: Rice Pest and Diseases	Weekly pest observations with station weather, 6 locations, 11 pests
NASA POWER weather data	NASA POWER	Daily temperature, humidity, rainfall and wind for the same 6 locations, 1981-2011


The NASA data was combined with the rice data to add an independent weather source for the
same locations and to cross-check the station readings.

Repository structure

nasa_daily/                       NASA POWER daily downloads, one file per location
RICE.csv                          original Kaggle dataset (unchanged)
nasa_weekly_all_locations.csv     NASA data aggregated to weekly values
RICE_merged.csv                   rice data combined with the weekly NASA weather
merge_data.py                     code that merges the two datasets
