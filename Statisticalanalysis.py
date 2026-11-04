# statistical_analysis.py
import os, itertools, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox

from visualization import load_insect_data

warnings.filterwarnings('ignore')
OUT_DIR = 'output_statistics'
os.makedirs(OUT_DIR, exist_ok=True)

WEATHER = {'MeanTemp': 'Temperature', 'NASA_RH': 'Humidity',
           'NASA_RF_mm': 'Rainfall', 'NASA_WS_kmph': 'Wind speed'}


def prepare(insects):
    insects = insects.copy()
    insects['MeanTemp'] = (insects['NASA_MaxT'] + insects['NASA_MinT']) / 2
    insects['LogCount'] = np.log1p(insects['Pest Value'])
    insects['Date'] = (pd.to_datetime(insects['Observation Year'].astype(int).astype(str) + '-01-01')
                       + pd.to_timedelta((insects['Standard Week'] - 1) * 7, unit='D'))
    return insects


def verdict(p):
    return 'significant' if p < 0.05 else 'not significant'


# ------------- 1. Descriptive statistics & normality -------------
def descriptive(insects):
    print('\n=== 1. DESCRIPTIVE STATISTICS & NORMALITY ===')
    d = stats.describe(insects['Pest Value'])
    print(f'Mean={d.mean:.2f}, Variance={d.variance:.2f}, '
          f'Skewness={d.skewness:.2f}, Kurtosis={d.kurtosis:.2f}')
    sample = insects['Pest Value'].sample(min(5000, len(insects)), random_state=1)
    for name, v in [('Raw count', sample), ('log(1+count)', np.log1p(sample))]:
        w, p = stats.shapiro(v)
        print(f'Shapiro-Wilk [{name}]: W={w:.3f}, p={p:.3g} -> '
              f'{"normal" if p > .05 else "NOT normal"}')
    return list(insects.groupby('Species')['Pest Value'].sum().nlargest(3).index)


# ------------- 2. Weather vs pest correlation -------------
def correlations(insects, top):
    print('\n=== 2. CORRELATION: WEATHER vs PEST COUNT ===')
    rows = []
    for sp in top:
        sub = insects[insects['Species'] == sp]
        for col, label in WEATHER.items():
            r, pr = stats.pearsonr(sub[col], sub['LogCount'])
            rho, ps = stats.spearmanr(sub[col], sub['LogCount'])
            rows.append([sp, label, r, pr, rho, ps, verdict(ps)])
    df = pd.DataFrame(rows, columns=['Species', 'Factor', 'Pearson_r', 'Pearson_p',
                                     'Spearman_rho', 'Spearman_p', 'Spearman result'])
    print(df.round(4).to_string(index=False))
    df.to_csv(os.path.join(OUT_DIR, 'correlations.csv'), index=False)


# ------------- 3. Hypothesis tests -------------
def hypothesis_tests(insects, top):
    print('\n=== 3. HYPOTHESIS TESTS ===')
    rows = []
    for sp in top:
        sub = insects[insects['Species'] == sp]

        # (a) Do stations differ?
        groups = [g['LogCount'].values for _, g in sub.groupby('Location') if len(g) > 5]
        lev, pl = stats.levene(*groups)
        f, pf = stats.f_oneway(*groups)
        h, pk = stats.kruskal(*groups)
        rows += [[sp, 'Stations: equal variances', 'Levene', lev, pl],
                 [sp, 'Stations: differ in mean count', 'One-way ANOVA', f, pf],
                 [sp, 'Stations: differ in count', 'Kruskal-Wallis', h, pk]]

        # (b) Does high humidity raise pest counts?
        cut = sub['NASA_RH'].median()
        high = sub.loc[sub['NASA_RH'] > cut, 'LogCount']
        low = sub.loc[sub['NASA_RH'] <= cut, 'LogCount']
        t, pt = stats.ttest_ind(high, low, equal_var=False)
        u, pu = stats.mannwhitneyu(high, low, alternative='greater')
        rows += [[sp, 'High vs low humidity', 'Welch t-test', t, pt],
                 [sp, 'High > low humidity', 'Mann-Whitney U', u, pu]]

    res = pd.DataFrame(rows, columns=['Species', 'Question', 'Test', 'Statistic', 'p_value'])
    res['Result'] = res['p_value'].apply(verdict)
    print(res.round(4).to_string(index=False))
    res.to_csv(os.path.join(OUT_DIR, 'hypothesis_tests.csv'), index=False)

    # (c) Chi-square: is an outbreak week associated with humidity / station?
    print('\nChi-square (outbreak = top 25% weekly counts):')
    rows = []
    for sp in top:
        sub = insects[insects['Species'] == sp].copy()
        sub['Outbreak'] = sub['Pest Value'] > sub['Pest Value'].quantile(.75)
        sub['Humidity'] = np.where(sub['NASA_RH'] > sub['NASA_RH'].median(), 'High', 'Low')
        for var in ['Humidity', 'Location']:
            chi, p, dof, _ = stats.chi2_contingency(pd.crosstab(sub[var], sub['Outbreak']))
            rows.append([sp, f'{var} vs outbreak', chi, dof, p, verdict(p)])
    print(pd.DataFrame(rows, columns=['Species', 'Variables', 'Chi2', 'dof', 'p_value',
                                      'Result']).round(4).to_string(index=False))


# ------------- 4. Time series forecasting -------------
def check_stationary(y, label):
    p = adfuller(y.dropna())[1]
    print(f'  ADF [{label}]: p={p:.4f} -> {"stationary" if p < .05 else "non-stationary"}')
    return p < 0.05


def forecast(insects, species, test_size=12):
    print(f'\n--- Forecast: {species} ---')
    sub = insects[insects['Species'] == species]
    y = sub.set_index('Date')['LogCount'].sort_index().resample('MS').mean()
    y = y.interpolate(limit_direction='both')
    if len(y) < 36:
        print('  Not enough months, skipped.'); return

    check_stationary(y, 'original')

    # Seasonal decomposition, then remove seasonality
    dec = seasonal_decompose(y, model='additive', period=12)
    dec.plot(); plt.gcf().set_size_inches(10, 8); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'decomposition_{species}.png'), dpi=200); plt.close()
    adjusted = y - dec.seasonal
    d = 0 if check_stationary(adjusted, 'seasonally adjusted') else 1
    if d:
        check_stationary(adjusted.diff().dropna(), 'after differencing')

    # ARIMA: choose order by AIC, evaluate on last 12 months
    train, test = adjusted.iloc[:-test_size], adjusted.iloc[-test_size:]
    best = None
    for p, q in itertools.product(range(4), range(4)):
        try:
            fit = ARIMA(train, order=(p, d, q)).fit()
            if best is None or fit.aic < best[1].aic:
                best = ((p, d, q), fit)
        except Exception:
            pass
    order, fit = best
    lb = acorr_ljungbox(fit.resid, lags=[10], return_df=True)['lb_pvalue'].iloc[0]
    print(f'  Best ARIMA{order}, AIC={fit.aic:.1f}, Ljung-Box p={lb:.3f} '
          f'({"good fit" if lb > .05 else "autocorrelation remains"})')

    fc = fit.get_forecast(steps=test_size)
    pred = fc.predicted_mean + dec.seasonal.loc[test.index]
    ci = fc.conf_int().add(dec.seasonal.loc[test.index], axis=0)
    actual = y.loc[test.index]
    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    naive = np.sqrt(np.mean((actual - y.iloc[-test_size - 1]) ** 2))
    print(f'  Test RMSE={rmse:.3f} | naive baseline RMSE={naive:.3f}')

    plt.figure(figsize=(10, 4))
    plt.plot(y.iloc[:-test_size][-48:], label='Train')
    plt.plot(actual, color='black', label='Actual')
    plt.plot(pred, color='red', label=f'ARIMA{order} forecast')
    plt.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], color='red', alpha=.15)
    plt.title(f'Forecast of {species} (monthly mean log1p count)')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'forecast_{species}.png'), dpi=200); plt.close()


def main():
    insects = prepare(load_insect_data())
    top = descriptive(insects)
    correlations(insects, top)
    hypothesis_tests(insects, top)
    for sp in top:
        forecast(insects, sp)
    print('\nOutputs saved in', OUT_DIR)

if __name__ == "__main__":
    main()
