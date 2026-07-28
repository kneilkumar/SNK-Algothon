from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, coint
import pandas as pd
import numpy as np
from johansen_engine import *

prices = pd.read_csv('prices.csv')
returns = np.log((prices/prices.shift(1)).dropna())


def adf_test():
    for instruments in prices.columns:
        adf_test_results = adfuller(returns[instruments], autolag='BIC', regression='c')
        print(f"instrument name: {instruments}\n"
              f"adf-stat: {adf_test_results[0]}\n"
              f"n-lags: {adf_test_results[2]}\n"
              f"p-val: {adf_test_results[1]}\n"
              f'Critical Values:')
        for key, value in adf_test_results[4].items():
            print('\t%s: %.3f' % (key, value))
        print('-------------------------------------')

        if (adf_test_results[1] <= 0.05) or (adf_test_results[0] <= adf_test_results[4]["1%"]) \
                or (adf_test_results[0] <= adf_test_results[4]["10%"]) or (
                adf_test_results[0] <= adf_test_results[4]["5%"]):
            if adf_test_results[1] <= 0.05:
                print("P-VAL IS SIG")

            if adf_test_results[0] <= adf_test_results[4]["1%"]:
                print("ADF-STAT MORE -VE THAN 1% LEVEL")

            if adf_test_results[0] <= adf_test_results[4]["10%"]:
                print("ADF-STAT MORE -VE THAN 10% LEVEL")

            if adf_test_results[0] <= adf_test_results[4]["5%"]:
                print("ADF-STAT MORE -VE THAN 5% LEVEL")
        print('-------------------------------------\n')


def acf_test():
    pass



import itertools
import dcor


n_instruments = 51
results = []


def create_dcorr_matrix():
    for i, j in itertools.combinations(range(n_instruments), 2):
        dcorr = dcor.distance_correlation(np.log(prices).iloc[:, i].values, np.log(prices).iloc[:, j].values)
        print(prices.columns.values[i])
        results.append({'instrument_1': i, 'instrument_2': j, 'distance_corr': dcorr})

    dcorr_df = pd.DataFrame(results).sort_values('distance_corr', ascending=False)
    dcorr_df.to_csv('log_price_dcorr_df.csv')


create_dcorr_matrix()


def create_dcorr_matrix_returns():
    prices_temp = np.log(prices.pct_change().dropna())
    for i, j in itertools.combinations(range(n_instruments), 2):
        dcorr = dcor.distance_correlation(prices_temp.iloc[:, i].values, prices_temp.iloc[:, j].values)
        print(prices_temp.columns.values[i])
        results.append({'instrument_1': i, 'instrument_2': j, 'distance_corr': dcorr})
    dcorr_df = pd.DataFrame(results).sort_values('distance_corr', ascending=False)
    dcorr_df.to_csv('dcorr_returns_df.csv')


def engle_granger(return_true):
    returns = prices.diff().dropna()
    if return_true:
        x = returns['AENO']
        y = returns['NWIG']
    else:
        x = prices['AENO']
        y = prices['NWIG']

    score_xy, pv_xy, crit_val_xy = coint(x, y)
    print(f"------------ {x.name} on {y.name} ---------------\n"
          f"score: {score_xy}\n"
          f"critical value: {crit_val_xy}\n"
          f"p-value: {pv_xy}\n")

    score_yx, pv_yx, crit_val_yx = coint(y,x)
    print(f"------------ {y.name} on {x.name} ---------------\n"
          f"score: {score_yx}\n"
          f"critical value: {crit_val_yx}\n"
          f"p-value: {pv_yx}\n")



import numpy as np
from numpy.linalg import eigh
from statsmodels.tsa.stattools import adfuller

# ---------------------------------------------------------
# 1. Extract top-k common factor(s) via PCA on returns
# ---------------------------------------------------------
def get_factors(returns, n_factors=1):
    """
    returns: nInst x nt (log returns)
    Returns factor time series (n_factors x nt) and loadings (nInst x n_factors)
    """
    X = (returns - returns.mean(axis=1, keepdims=True)) / returns.std(axis=1, keepdims=True)
    cov = np.cov(X)
    eigvals, eigvecs = eigh(cov)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    loadings = eigvecs[:, :n_factors]          # nInst x n_factors
    factors = loadings.T @ X                    # n_factors x nt, factor return series

    return factors, loadings, eigvals


# ---------------------------------------------------------
# 2. Regress each instrument's returns on the factor(s),
#    keep the residual (idiosyncratic) return series
# ---------------------------------------------------------
def get_residual_returns(returns, factors):
    """
    returns: nInst x nt
    factors: n_factors x nt
    Returns: residual_returns (nInst x nt), betas (nInst x n_factors)
    """
    nInst, nt = returns.shape
    n_factors = factors.shape[0]

    F = factors.T  # nt x n_factors
    F = np.hstack([np.ones((nt, 1)), F])  # add intercept

    residuals = np.zeros_like(returns)
    betas = np.zeros((nInst, n_factors))

    for i in range(nInst):
        y = returns[i, :]
        coef, _, _, _ = np.linalg.lstsq(F, y, rcond=None)
        y_hat = F @ coef
        residuals[i, :] = y - y_hat
        betas[i, :] = coef[1:]  # drop intercept

    return residuals, betas


# ---------------------------------------------------------
# 3. Build "synthetic idiosyncratic price" series
#    (cumulative sum of residual returns) — this is the
#    Avellaneda-Lee style construction: if the factor
#    exposure is properly stripped out, this integrated
#    residual should itself be mean-reverting (stationary)
# ---------------------------------------------------------
def get_residual_levels(residual_returns):
    return np.cumsum(residual_returns, axis=1)  # nInst x nt


# ---------------------------------------------------------
# 4a. Single-name check: is each instrument's OWN residual
#     level series mean-reverting? (no pairing needed)
# ---------------------------------------------------------
def screen_single_name_reversion(residual_levels, adf_pval_thresh=0.05):
    nInst, nt = residual_levels.shape
    results = []
    for i in range(nInst):
        series = residual_levels[i, :]
        stat, pval, *_ = adfuller(series, regression='c', autolag='BIC')

        # fit AR(1) on the residual level series to get half-life
        y = series[1:]
        x = series[:-1]
        phi = np.polyfit(x, y, 1)[0]
        half_life = np.log(0.5) / np.log(phi) if 0 < phi < 1 else np.inf

        results.append({
            'instrument': i,
            'adf_pval': pval,
            'phi': phi,
            'half_life': half_life,
            'stationary': pval < adf_pval_thresh
        })
    return results


# ---------------------------------------------------------
# 4b. Pairwise check: cointegration between RESIDUAL levels
#     (i.e. after common factor is stripped, do any two
#     instruments still share an idiosyncratic stochastic trend)
# ---------------------------------------------------------
def screen_residual_pairs_johansen(residual_levels, johansen_test_fn):
    """
    johansen_test_fn: your existing Johansen wrapper from the pairs engine —
    pass it in so this reuses the same test/threshold logic you already built,
    just pointed at residual_levels instead of raw prices.
    """
    nInst = residual_levels.shape[0]
    results = []
    for i in range(nInst):
        for j in range(i + 1, nInst):
            res = johansen_test_fn(
    residual_levels, instrument_names=prices.columns.values, group_size=2,
    train_frac=0.67, fwer_alpha=0.05, min_cycles=3.5,
    n_null_sims=5000, max_tests=25_000, seed=42,
)
            results.append({'pair': (i, j), **res})
    return results


# ---------------------------------------------------------
# Usage sketch

pricesT = prices.T
returns = np.diff(np.log(pricesT), axis=1)
# prices2 = np.array(prices)
#
# factors, loadings, eigvals = get_factors(returns, n_factors=1)
# residual_returns, betas = get_residual_returns(returns, factors)
# residual_levels = get_residual_levels(residual_returns)
#
# single_name_results = screen_single_name_reversion(residual_levels)
# pair_results = johansen_engine(
#     residual_levels, instrument_names=prices.columns.values, group_size=3,
#     train_frac=0.67, fwer_alpha=0.05, min_cycles=3.5,
#     n_null_sims=5000, max_tests=25_000, seed=42)


# ---------------------------------------------------------
# 1. Derive rolling window length from measured half-life
# ---------------------------------------------------------
def window_from_half_life(half_life, n_independent_samples=8):
    """
    half_life: estimated AR(1) half-life of the spread (in days)
    n_independent_samples: how many "independent" observations
                            you want your rolling mean/std to rest on

    Rationale: autocorrelation time of a mean-reverting series is
    roughly 2x its half-life -- points closer together than that
    aren't providing independent information about the mean/std.
    """
    autocorr_time = 2 * half_life
    window = int(np.round(n_independent_samples * autocorr_time))
    return window


# example: half_life ~5.8 (train) or 2.5 (holdout) -- pick a
# conservative (larger) one, or average, but decide deliberately
window_len = window_from_half_life(half_life=5.8, n_independent_samples=8)
print(f"Suggested rolling window: {window_len} days")


# ---------------------------------------------------------
# 2. Build the spread + z-score series
# ---------------------------------------------------------
def compute_spread_zscore(price_hett, price_ulxy, beta, window):
    spread = price_hett - beta * price_ulxy  # match your engine's sign convention

    z = np.full_like(spread, np.nan)
    for t in range(window, len(spread)):
        window_slice = spread[t - window:t]
        mu = window_slice.mean()
        sigma = window_slice.std()
        z[t] = (spread[t] - mu) / sigma if sigma > 0 else 0.0

    return spread, z


# ---------------------------------------------------------
# 3. Simulate positions + PL for a given (entry, exit) threshold pair
#    using YOUR actual scoring function, not generic Sharpe
# ---------------------------------------------------------
def simulate_strategy(price_hett, price_ulxy, beta, z, entry_z, exit_z,
                       cap_dollars=10_000, fee_bps=10):
    n = len(z)
    pos_hett = np.zeros(n, dtype=int)
    pos_ulxy = np.zeros(n, dtype=int)
    in_position = 0  # -1 short spread, 0 flat, +1 long spread

    for t in range(1, n):
        if np.isnan(z[t]):
            continue

        # entry logic
        if in_position == 0:
            if z[t] > entry_z:
                in_position = -1   # spread too wide -> short spread
            elif z[t] < -entry_z:
                in_position = 1    # spread too low -> long spread
        # exit logic (hysteresis: exit threshold looser than entry)
        else:
            if abs(z[t]) < exit_z:
                in_position = 0

        if in_position != 0:
            # target dollar exposure per leg, scaled by direction and hedge ratio
            dollar_hett = in_position * cap_dollars
            dollar_ulxy = -in_position * beta * cap_dollars  # preserve ratio

            # convert to shares, clip to cap, round to integer
            n_hett = int(np.clip(dollar_hett / price_hett[t], -cap_dollars/price_hett[t], cap_dollars/price_hett[t]))
            n_ulxy = int(np.clip(dollar_ulxy / price_ulxy[t], -cap_dollars/price_ulxy[t], cap_dollars/price_ulxy[t]))
        else:
            n_hett, n_ulxy = 0, 0

        pos_hett[t] = n_hett
        pos_ulxy[t] = n_ulxy

    # PL calculation (simplified -- daily mark-to-market plus fees on position changes)
    pl = np.zeros(n)
    for t in range(1, n):
        price_change_hett = price_hett[t] - price_hett[t-1]
        price_change_ulxy = price_ulxy[t] - price_ulxy[t-1]
        pl[t] = pos_hett[t-1] * price_change_hett + pos_ulxy[t-1] * price_change_ulxy

        # fees on position change (both legs, 10bps notional)
        turnover_hett = abs(pos_hett[t] - pos_hett[t-1]) * price_hett[t]
        turnover_ulxy = abs(pos_ulxy[t] - pos_ulxy[t-1]) * price_ulxy[t]
        pl[t] -= (turnover_hett + turnover_ulxy) * (fee_bps / 10000)

    return pl, pos_hett, pos_ulxy


# ---------------------------------------------------------
# 4. Your actual scoring function
# ---------------------------------------------------------
def score(pl):
    mu = pl.mean()
    sigma = pl.std()
    if sigma == 0:
        return mu
    SR = np.sqrt(250) * mu / sigma
    if mu < 0:
        return mu
    return mu * (SR**2) / (SR**2 + 1)


# ---------------------------------------------------------
# 5. Grid search entry/exit thresholds on TRAIN, check on HOLDOUT
# ---------------------------------------------------------
def threshold_grid_search(price_hett_train, price_ulxy_train, beta, window,
                           entry_grid=(0.5,1.5, 2.0, 2.5, 3.0),
                           exit_grid=(0.25, 0.5, 0.75)):

    spread_train, z_train = compute_spread_zscore(price_hett_train, price_ulxy_train, beta, window)

    results = []
    for entry_z in entry_grid:
        for exit_z in exit_grid:
            if exit_z >= entry_z:
                continue
            pl, _, _ = simulate_strategy(price_hett_train, price_ulxy_train, beta, z_train, entry_z, exit_z)
            results.append({'entry_z': entry_z, 'exit_z': exit_z, 'score': score(pl)})

    results.sort(key=lambda r: r['score'], reverse=True)
    return results


# usage:
price_hett_train = pricesT.loc['HETT', :].reset_index(drop=True)
price_ulxy_train = pricesT.loc['ULXY', :].reset_index(drop=True)
res2 = threshold_grid_search(price_hett_train, price_ulxy_train, beta=1.65382303, window=window_len)
best = res2[0]
print(best)
#
# then re-simulate with best['entry_z'], best['exit_z'] on the HOLDOUT prices
# and check score(pl_holdout) isn't drastically worse than score(pl_train) --
# if it collapses, the threshold was overfit, back off to a rounder number


