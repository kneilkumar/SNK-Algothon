"""
example_usage.py

How to point johansen_engine at your actual 51-instrument price history.
"""

import numpy as np
import pandas as pd
from johansen_engine import johansen_engine, load_price_matrix

# ---------------------------------------------------------------------------
# 1) Load your price data as an (nInst, nt) array of LEVELS.
#
#    If you've got the typical competition prices.txt (rows = time steps,
#    columns = instruments, whitespace-delimited), this helper transposes it
#    into the shape this engine expects:
#
#        prices = load_price_matrix("prices.txt")
#
#    Otherwise just build the array yourself, e.g. from a DataFrame:
#
#        prices = df[instrument_columns].values.T
#
#    IMPORTANT: pass your full history in here and let train_frac do the
#    chronological split -- don't pre-split it yourself, since the engine's
#    holdout confirmation step needs to do its own independent re-fit on
#    the untouched slice.
# ---------------------------------------------------------------------------

prices = pd.read_csv('prices.csv').T  # <-- replace with your actual data
instrument_names = prices.index.values  # or your real tickers

# ---------------------------------------------------------------------------
# 2) Run the screen. group_size=2 replicates the pairs spec exactly.
#    Bumping group_size to 3 or 4 is the "basket" extension -- same
#    function, same guarantees, just a different combinatorial fan-out
#    (C(51,3) = 20,825; C(51,4) = 249,900, hence the max_tests safety cap).
# ---------------------------------------------------------------------------

# out = johansen_engine(
#     prices,
#     instrument_names=instrument_names,
#     group_size=2,          # 2 = pairs, 3 = triplets, 4 = quadruplets, ...
#     train_frac=0.8,        # matches your 0.67:0.33 train:test competition split
#     fwer_alpha=0.05,        # family-wise error rate to control via Bonferroni
#     min_cycles=3.5,         # require >= 3.5 half-lives inside the window
#     n_null_sims=5000,       # Monte Carlo draws for the Bonferroni threshold
#     holdout_alpha_level="95",
#     seed=42,
# )



# If you want triplets too (only worth doing if pairs come up short):
#
out3 = johansen_engine(
    prices, instrument_names=instrument_names, group_size=2,
    train_frac=0.8, fwer_alpha=0.05, min_cycles=3.5,
    n_null_sims=5000, max_tests=25_000, seed=42,
)

print(out3["meta"])
print(out3["candidates"])

# ---------------------------------------------------------------------------
# 3) The 'candidates' DataFrame is your short list for getMyPosition():
#    each surviving row gives you the normalized cointegrating vector
#    (hedge ratios, first instrument's coefficient = 1), the AR(1)
#    half-life on both train and holdout, and the holdout pass/fail flags
#    that make it a *confirmed* candidate rather than just a train-set
#    artifact.
# ---------------------------------------------------------------------------

for _, row in out3["candidates"].iterrows():
    print(
        f"{row['combo']}: hedge_ratios={row['coint_vector_normalized']}, "
        f"half_life_train={row['half_life_train']:.1f}, "
        f"half_life_holdout={row['holdout_half_life']:.1f}"
    )

import numpy as np
from numpy.linalg import eigh


def check_common_factor(prices, use_returns=True):
    if use_returns:
        data = np.diff(np.log(prices), axis=1)  # log returns, nInst x (nt-1)
    else:
        data = prices

    # standardize each instrument (zero mean, unit variance) before PCA
    # so no single high-variance instrument dominates the covariance matrix
    X = (data - data.mean(axis=1, keepdims=True)) / data.std(axis=1, keepdims=True)

    # covariance across instruments (nInst x nInst)
    cov = np.cov(X)

    eigvals, eigvecs = eigh(cov)  # ascending order
    eigvals = eigvals[::-1]        # flip to descending
    eigvecs = eigvecs[:, ::-1]

    explained = eigvals / eigvals.sum()
    cumulative = np.cumsum(explained)

    print("Top 10 eigenvalues (explained variance ratio):")
    for i in range(10):
        print(f"  PC{i+1}: {explained[i]:.4f}   cumulative: {cumulative[i]:.4f}")

    print(f"\nPC1 alone explains {explained[0]*100:.1f}% of variance")
    print(f"Top 3 PCs explain {cumulative[2]*100:.1f}% of variance")

    return eigvals, eigvecs, explained


prices = pd.read_csv('prices.csv')  # shape (51, nt)
eigvals, eigvecs, explained = check_common_factor(prices, use_returns=True)
