import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

OUT_DIR = 'output_visualizations'
os.makedirs(OUT_DIR, exist_ok=True)
sns.set_theme(style='whitegrid')

# Cleaner labels for the species names in the CSV
SPECIES_LABELS = {
    'Brownplanthopper': 'Brown planthopper',
    'Caseworm': 'Caseworm',
    'Gallmidge': 'Gall midge',
    'Greenleafhopper': 'Green leafhopper',
    'LeafFolder': 'Leaf folder',
    'Miridbug': 'Mirid bug',
    'Whitebackedplanthopper': 'White-backed planthopper',
    'Yellowstemborer': 'Yellow stem borer',
    'ZigZagleafhopper': 'Zig-zag leafhopper',
}

STATION_COORDS = {
    'Cuttack': (20.46, 85.88),
    'Ludhiana': (30.90, 75.86),
    'Maruteru': (16.63, 81.72),
    'Raipur': (21.25, 81.63),
    'Rajendranagar': (17.32, 78.40),
}


def load_insect_data():
    data = pd.read_csv('RICE_merged_cleaned.csv')
    insects = data[data['Collection Type'] == 'Number/Light trap'].copy()
    # a few (year, week, species, station) combinations appear twice; keep the first
    insects = insects.drop_duplicates(
        subset=['Observation Year', 'Standard Week', 'PEST NAME', 'Location'])
    insects['Species'] = insects['PEST NAME'].map(SPECIES_LABELS)
    return insects


def save_figure(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=200, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------- 1. Heatmap
def plot_location_species_heatmap(insects):
    table = insects.pivot_table(index='Location', columns='Species',
                                values='Pest Value', aggfunc='mean')
    table = table.loc[table.sum(axis=1).sort_values(ascending=False).index,
                      table.sum(axis=0).sort_values(ascending=False).index]

    plt.figure(figsize=(11, 5.5))
    ax = sns.heatmap(table, cmap='YlOrRd', linewidths=0.5, linecolor='white',
                     norm=plt.matplotlib.colors.LogNorm(vmin=1, vmax=table.max().max()),
                     annot=table.round(0), fmt='g', annot_kws={'size': 8},
                     cbar_kws={'label': 'Mean weekly count per trap (log colour scale)'})
    ax.set_facecolor('lightgrey')          # grey = species not recorded at that station
    plt.title('Mean Weekly Light-Trap Count of Each Pest Species at Each Station\n'
              '(grey = not recorded)')
    plt.xlabel('Pest species')
    plt.ylabel('Station')
    plt.xticks(rotation=40, ha='right')
    save_figure('1_location_species_heatmap.png')


# ---------------------------------------------------------------- 2. Ranking
def plot_species_ranking(insects):
    summary = insects.groupby('Species')['Pest Value'].agg(['sum', 'mean'])
    summary = summary.sort_values('sum', ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, column, title in [(axes[0], 'sum', 'Total count (all stations, all years)'),
                              (axes[1], 'mean', 'Mean weekly count per record')]:
        values = summary[column]   # both panels use the same (total-based) order
        # highlight the top species, others in a neutral colour
        colours = ['firebrick' if i == 0 else 'steelblue' for i in range(len(values))]
        ax.barh(values.index, values.values, color=colours)
        ax.set_title(title)
        ax.set_xlabel('Light-trap count')
        for i, v in enumerate(values.values):
            ax.text(v, i, f' {v:,.0f}', va='center', fontsize=8)
        ax.set_xlim(0, values.max() * 1.18)
    axes[0].invert_yaxis()         # largest at the top (axis is shared)
    axes[0].set_ylabel('Pest species')
    fig.suptitle('Dominant Rice Pest Species (light-trap catches)', fontsize=13)
    save_figure('2_species_ranking.png')


# ------------------------------------------------------------- 3. Stacked bar
def plot_stacked_distribution(insects, top_n=6):
    top_species = insects.groupby('Species')['Pest Value'].sum().nlargest(top_n).index
    grouped = insects.copy()
    grouped['Group'] = grouped['Species'].where(grouped['Species'].isin(top_species), 'Others')
    table = grouped.pivot_table(index='Location', columns='Group',
                                values='Pest Value', aggfunc='sum', fill_value=0)
    table = table[list(top_species) + ['Others']]
    table = table.loc[table.sum(axis=1).sort_values(ascending=False).index]
    colours = sns.color_palette('tab10', top_n) + [(0.7, 0.7, 0.7)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    table.plot(kind='bar', stacked=True, ax=axes[0], color=colours, width=0.75)
    axes[0].set_title('Total pest quantity')
    axes[0].set_ylabel('Total light-trap count')
    # same data as percentages, so composition can be compared between stations
    share = table.div(table.sum(axis=1), axis=0) * 100
    share.plot(kind='bar', stacked=True, ax=axes[1], color=colours, width=0.75, legend=False)
    axes[1].set_title('Share of each species (%)')
    axes[1].set_ylabel('Percentage of station total')
    for ax in axes:
        ax.set_xlabel('Station')
        ax.tick_params(axis='x', rotation=30)
        ax.grid(False, axis='x')
    axes[0].legend(title='Species', fontsize=8)
    fig.suptitle(f'Pest Composition by Station (top {top_n} species, rest grouped as Others)',
                 fontsize=13)
    save_figure('3_stacked_pest_distribution.png')


# ------------------------------------------------------------ 4. Seasonal trend
def plot_seasonal_trends(insects, top_n=5):
    day_of_year = (insects['Standard Week'] - 1) * 7 + 4
    insects = insects.assign(Month=(pd.Timestamp('2001-01-01')
                                    + pd.to_timedelta(day_of_year, unit='D')).dt.month)

    top_species = insects.groupby('Species')['Pest Value'].sum().nlargest(top_n).index
    monthly = (insects[insects['Species'].isin(top_species)]
               .groupby(['Month', 'Species'])['Pest Value'].mean().reset_index())

    plt.figure(figsize=(11, 6))
    sns.lineplot(data=monthly, x='Month', y='Pest Value', hue='Species',
                 hue_order=list(top_species), marker='o', linewidth=2)
    plt.yscale('log')   # species differ by orders of magnitude
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.title('Seasonal Pattern of the Top 5 Pest Species\n'
              '(average over all stations and years)')
    plt.xlabel('Month (approximated from Standard Week)')
    plt.ylabel('Mean weekly light-trap count (log scale)')
    plt.legend(title='Species', bbox_to_anchor=(1.02, 1), loc='upper left')
    save_figure('4_seasonal_trends.png')


# ------------------------------------------------- 5. Weather vs pest (optional)
def plot_weather_vs_pests(insects, top_n=3):
    insects = insects.assign(MeanTemp=(insects['MaxT'] + insects['MinT']) / 2,
                             LogCount=np.log10(1 + insects['Pest Value']))
    top_species = insects.groupby('Species')['Pest Value'].sum().nlargest(top_n).index
    weather = [('MeanTemp', 'Mean temperature (°C)'),
               ('RH1(%)', 'Morning relative humidity (%)')]

    fig, axes = plt.subplots(2, top_n, figsize=(5 * top_n, 8), sharey='col')
    for col, species in enumerate(top_species):
        subset = insects[insects['Species'] == species]
        for row, (column, label) in enumerate(weather):
            ax = axes[row, col]
            sns.regplot(data=subset, x=column, y='LogCount', ax=ax,
                        scatter_kws={'alpha': 0.15, 's': 10},
                        line_kws={'color': 'red'})
            rho = subset[column].corr(subset['LogCount'], method='spearman')
            ax.set_title(f'{species}  (Spearman ρ = {rho:.2f})', fontsize=10)
            ax.set_ylim(bottom=-0.1)   # counts cannot be negative; stops the line dipping below 0
            ax.set_xlabel(label)
            ax.set_ylabel('log10(1 + weekly count)' if col == 0 else '')
    fig.suptitle('Weather vs Pest Count (association only, not proof of cause)', fontsize=13)
    save_figure('5_weather_vs_pests.png')


# ------------------------------------------------------ 6. Interactive map (optional)
def make_pest_map(insects):
    stats = (insects.groupby(['Location', 'Species'])['Pest Value']
             .agg(Mean='mean', Total='sum')
             .join(insects.groupby(['Location', 'Species'])['Observation Year'].nunique()
                   .rename('Years')).reset_index())
    stats = stats[stats['Location'].isin(STATION_COORDS)]
    stats['Lat'] = stats['Location'].map(lambda s: STATION_COORDS[s][0])
    stats['Lon'] = stats['Location'].map(lambda s: STATION_COORDS[s][1])
    species_order = list(insects.groupby('Species')['Pest Value'].sum()
                         .sort_values(ascending=False).index)

    fig = go.Figure()
    for i, species in enumerate(species_order):
        sub = stats[stats['Species'] == species]
        fig.add_trace(go.Scattergeo(
            lon=sub['Lon'], lat=sub['Lat'], name=species, visible=(i == 0),
            customdata=sub[['Location', 'Species', 'Mean', 'Total', 'Years']].values,
            hovertemplate=('<b>%{customdata[0]}</b><br>Species: %{customdata[1]}<br>'
                           'Mean weekly count: %{customdata[2]:,.0f}<br>'
                           'Total count: %{customdata[3]:,.0f}<br>'
                           'Years observed: %{customdata[4]}<extra></extra>'),
            marker=dict(size=12 + 45 * np.sqrt(sub['Mean'] / sub['Mean'].max()),
                        color=sub['Mean'], colorscale='YlOrRd',
                        colorbar=dict(title='Mean weekly<br>count'),
                        line=dict(width=1, color='black'), opacity=0.85)))

    # station names (always visible)
    names = pd.Series(STATION_COORDS)
    fig.add_trace(go.Scattergeo(
        lon=[v[1] for v in names], lat=[v[0] for v in names], text=list(names.index),
        mode='text', textposition='top center', hoverinfo='skip', showlegend=False))

    buttons = []
    for i, species in enumerate(species_order):
        visible = [j == i for j in range(len(species_order))] + [True]
        buttons.append(dict(label=species, method='update', args=[{'visible': visible}]))
    fig.update_layout(
        title='Rice Pest Activity by Research Station (select a species; bubble size and '
              'colour = mean weekly light-trap count)',
        updatemenus=[dict(buttons=buttons, x=0.01, y=0.99, xanchor='left', yanchor='top')],
        margin=dict(l=0, r=0, t=60, b=0), height=650)
    fig.update_geos(scope='asia', lonaxis_range=[68, 92], lataxis_range=[8, 35],
                    showcountries=True, countrycolor='grey', showland=True,
                    landcolor='whitesmoke', showocean=True, oceancolor='aliceblue')
    fig.write_html(os.path.join(OUT_DIR, '6_pest_map.html'), include_plotlyjs=True)


def main():
    insects = load_insect_data()
    plot_location_species_heatmap(insects)
    plot_species_ranking(insects)
    plot_stacked_distribution(insects)
    plot_seasonal_trends(insects)
    plot_weather_vs_pests(insects)
    make_pest_map(insects)
    print('Saved all visualizations in', OUT_DIR)


if __name__ == '__main__':
    main()