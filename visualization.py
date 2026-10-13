
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('RICE_merged_cleaned.csv')

# 1. Correlation heatmap
weather_cols = ['Pest Value', 'MaxT', 'MinT', 'RH1(%)', 'RH2(%)',
                'RF(mm)', 'WS(kmph)', 'SSH(hrs)', 'EVP(mm)']

corr_matrix = data[weather_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            vmin=-1, vmax=1, linewidths=0.5)
plt.title('Correlation Heatmap: Pest Outbreaks vs Environmental Factors')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()

# 2. Total pests at each location
location_data = data.groupby('Location')['Pest Value'].sum().reset_index()
location_data = location_data.sort_values('Pest Value', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=location_data, x='Location', y='Pest Value', hue='Location', legend=False, palette='viridis')
plt.title('Total Pest Incidence Across Research Stations')
plt.xlabel('Research Station Location')
plt.ylabel('Total Pest Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('pests_by_location.png')
plt.show()

# 3. Weekly pest trend
weekly_data = data.groupby('Standard Week')['Pest Value'].mean().reset_index()
weekly_data = weekly_data.sort_values('Standard Week')

plt.figure(figsize=(12, 6))
sns.lineplot(data=weekly_data, x='Standard Week', y='Pest Value',
             marker='o', color='darkred', linewidth=2.5)
plt.title('Seasonal Pest Outbreak Trajectory')
plt.xlabel('Standard Week')
plt.ylabel('Average Pest Count per Station')
plt.axvspan(28, 42, color='orange', alpha=0.2,
            label='Peak Outbreak Window (Weeks 28-42)')
plt.legend()
plt.tight_layout()
plt.savefig('seasonal_outbreak_trajectory.png')
plt.show()

# 4. Most common pests
top_pests = data.groupby('PEST NAME')['Pest Value'].sum().reset_index()
top_pests = top_pests.sort_values('Pest Value', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=top_pests, x='Pest Value', y='PEST NAME', hue='PEST NAME', legend=False, palette='magma')
plt.title('Total Outbreak Count by Rice Pest Species')
plt.xlabel('Total Recorded Count')
plt.ylabel('Pest Species')
plt.tight_layout()
plt.savefig('top_pest_species.png')
plt.show()

# 5. Humidity vs temperature
filtered_data = data[data['Pest Value'] > 0]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=filtered_data, x='RH1(%)', y='MinT',
                hue='Location', size='Pest Value',
                sizes=(20, 500), alpha=0.7)
plt.title('Pest Outbreaks: Humidity vs Minimum Temperature')
plt.xlabel('Morning Relative Humidity (%)')
plt.ylabel('Minimum Temperature (°C)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('humidity_temp_scatter.png')
plt.show()
