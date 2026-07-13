import time
import numpy as np
from johansen_engine import johansen_engine
import pandas as pd

rng = np.random.default_rng(7)
n_inst = 51
n_t = 500
tickers = pd.read_csv('prices.csv').columns.values


# Base: correlated random walks sharing a common market factor (drift
# contamination) -- this is exactly the kind of thing that would fool a
# naive dcorr filter.
market_factor = np.cumsum(rng.normal(0.02, 1.0, n_t))
prices = np.zeros((n_inst, n_t))
for i in range(n_inst):
    idio = np.cumsum(rng.normal(0, 1.0, n_t))
    beta_mkt = rng.uniform(0.3, 1.2)
    prices[i] = 100 + beta_mkt * market_factor + idio

# Engineer 3 genuinely cointegrated pairs by overwriting one leg of the
# pair as a linear combo of the other plus stationary noise.
true_pairs = [(0, 1), (10, 11), (30, 31)]
for a, b in true_pairs:
    hedge = rng.uniform(0.5, 2.0)
    stat_noise = np.zeros(n_t)
    phi_true = 0.85
    eps = rng.normal(0, 1.0, n_t)
    for t in range(1, n_t):
        stat_noise[t] = phi_true * stat_noise[t - 1] + eps[t]
    prices[b] = hedge * prices[a] + stat_noise

names = tickers

t0 = time.time()
out = johansen_engine(
    prices,
    instrument_names=names,
    group_size=2,
    train_frac=0.67,
    fwer_alpha=0.05,
    min_cycles=3.0,
    n_null_sims=3000,
    seed=1,
    verbose=True,
)
print(f"elapsed: {time.time()-t0:.1f}s")

print("\n--- final candidates ---")
cols = ["combo", "trace_stat", "p_value_sim", "pass_bonferroni",
        "half_life_train", "holdout_trace_stat", "holdout_half_life",
        "final_candidate"]
print(out["candidates"][cols] if len(out["candidates"]) else "none")

print("\ntrue engineered pairs:", [(names[a], names[b]) for a, b in true_pairs])

print("\ntop 10 by trace stat overall:")
print(out["all_results"][["combo", "trace_stat", "pass_95_tabulated",
                           "pass_bonferroni"]].head(10))
