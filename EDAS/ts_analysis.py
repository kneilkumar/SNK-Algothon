import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import false_discovery_control
from scipy import stats
from statsmodels.tsa.ar_model import AutoReg

from ADF_ACF_tests import adfuller



prices = pd.read_csv('prices.csv')
returns = prices.pct_change().dropna()

# for inst in returns.columns:
#     fig, ax = plt.subplots(figsize=(8,5))
#     plot_pacf(returns[inst], lags=20, ax=ax, method='ywadjusted')
#
#     plt.xlabel("Lags")
#     plt.ylabel("Partial Autocorrelation")
#     plt.title(f"Partial Autocorrelation Function (PACF) for {inst} returns")
#     plt.show()
#
# for inst in returns.columns:
#     fig2, ax2 = plt.subplots(figsize=(8,5))
#     plot_acf(returns[inst], lags=20, ax=ax2)
#     plt.xlabel("Lags")
#     plt.ylabel("Autocorrelation")
#     plt.title(f"Autocorrelation Function (ACF) for {inst} returns")
#     plt.show()

return_means = returns.mean()
print(return_means)

mean_model = 0

for inst in returns.columns:
    y = returns[inst]

    t_stat, p_value = stats.ttest_1samp(y, 0)

    if p_value < 0.05:
        print("Reject the null hypothesis: Significant difference exists between the group means.")
        print(f"T-statistic: {t_stat:.4f}")
        print(f"P-value: {p_value:.4f}")
        x = return_means[inst]
    else:
        x = 0
    residuals = y - x
    res_squared = np.square(residuals)
    lb_res = acorr_ljungbox(res_squared, lags=[5,10,15,20], return_df=True)

    p_val_list = lb_res['lb_pvalue'].tolist()
    BH_pvals = false_discovery_control(p_val_list)
    if any(BH_pvals < 0.05):
        print(f" adjusted p_vals for {inst}: {BH_pvals}")


garch_candidates = returns[['HUXZ', 'ACAC']]

from arch import arch_model
model_list = []
for candidate in garch_candidates.columns:
    returns_series = garch_candidates[candidate]
    model = arch_model(returns_series, mean='constant', vol='EGARCH', p=1, q=1, rescale=False)
    model_fitted = model.fit(update_freq=5)
    print(model_fitted.summary())
    model_list.append(model_fitted)

    garch_resid = model_fitted.std_resid**2

    lb_vol_res = acorr_ljungbox(garch_resid, lags=[5,10,15,20], return_df=True)
    print(f"-------------\n{candidate}")
    print(lb_vol_res)
    bh_adj = false_discovery_control(lb_vol_res['lb_pvalue'].tolist())
    print(bh_adj)
    print("--------------")


huxz_model = np.array(model_list[0].conditional_volatility)
acac_model = np.array(model_list[1].conditional_volatility)

hedge_data = np.array(prices[['HUXZ', 'ACAC']].iloc[1:]).transpose()



# def _ols_hedge(a, b):
#     """Closed-form OLS hedge ratio & intercept for a ~ hedge*b + const."""
#     b_mean = b.mean()
#     a_mean = a.mean()
#     var = np.dot(b - b_mean, b - b_mean)
#     if var < 1e-9:
#         return None, None
#     hedge = np.dot(b - b_mean, a - a_mean) / var
#     const = a_mean - hedge * b_mean
#     return hedge, const

# for t in range(1, 750):
#     # price history up to and including t, e.g. if t=500, gets first 500 days
#     hedge_dataSoFar = hedge_data[:, :t]
#     curPrices = hedge_dataSoFar[:, -1]
#
#     # trading loop, do not do it on the very last day of the test
#     if t < 750:
#         nInst, nDays = hedge_dataSoFar.shape
#         positions = np.zeros(nInst, dtype=int)
#         prices_today = hedge_dataSoFar[:, -1]
#
#         # ---------------- Pair trades ----------------
#         if nDays >= 42:
#             pa = hedge_dataSoFar[0]
#             pb = hedge_dataSoFar[1]
#             w_hedge = min(300, nDays)
#             hedge, const = _ols_hedge(pa[-w_hedge:], pb[-w_hedge:])
#             rolling_hedge_ratio.append(hedge)
#         else:
#             rolling_hedge_ratio.append(0)

print(len(model_list[0].conditional_volatility), len(hedge_data[0]))
beta = np.array((acac_model*hedge_data[1])/(huxz_model*hedge_data[0]))
spread = hedge_data[0] - beta*hedge_data[1]

adf_test_results = adfuller(spread, autolag='BIC', regression='c')
print(f"adf-stat: {adf_test_results[0]}\n"
      f"n-lags: {adf_test_results[2]}\n"
      f"p-val: {adf_test_results[1]}\n"
      f'Critical Values:')
for key, value in adf_test_results[4].items():
    print('\t%s: %.3f' % (key, value))
print('-------------------------------------')

OU = AutoReg(spread, lags=1, trend='c')
OU_fit = OU.fit()
print(f"Instrument: {spread} OU summary:\n\n")
print(OU_fit.summary())
print(f"half life: {np.log(0.5) / np.log(OU_fit.params[1])}")
print(OU_fit.params)

print(np.mean(model_list[0].conditional_volatility + model_list[1].conditional_volatility))
