"""
Independent train/holdout ADF verification for cointegration candidates.

WHAT THIS ANSWERS
------------------
For each candidate group of tickers, this checks TWO things separately,
and reports them separately -- it does NOT just run one ADF test on the
pooled train+holdout series:

  1. TRAIN-ONLY validation:
       - Fit the hedge ratio (Johansen) using ONLY the first TRAIN_DAYS
         days.
       - Run ADF on the resulting TRAIN residual/spread.
  2. HOLDOUT validation, OUT OF SAMPLE:
       - Apply the SAME hedge ratio fitted on train (never refit on
         holdout) to the last HOLDOUT_DAYS days.
       - Run ADF independently on that HOLDOUT residual/spread.

A candidate only counts as genuinely validated if BOTH tests pass
independently. If you instead fit + test on the full pooled window, you
can get a "pass" purely because the in-sample fit is flattering the
pooled ADF stat -- that is NOT the same claim as "this hedge ratio,
estimated on train, produced a stationary residual on unseen holdout
data."

For reference, this script also reports a POOLED ADF test (fit + test
on all TRAIN_DAYS+HOLDOUT_DAYS days together) so you can directly see
whether/how much it diverges from the honest train/holdout split -- a
big gap between "pooled looks great" and "train+holdout independently
both pass" is itself informative about overfitting.

WHAT YOU NEED TO EDIT
----------------------
  - TICKER_TO_IDX: map each ticker string to its column index in
    `prcSoFar` (0-50).
  - CANDIDATES: list of ticker groups (2 tickers = pair, 3+ = Johansen
    on that many legs). Put your 8 candidates here.
  - TRAIN_DAYS / HOLDOUT_DAYS: must match how you originally split
    (defaults to 335 / 165, as used for HETT/ULXY).
  - N_TOTAL_TESTS: total number of pairs/relationships you screened
    across your whole search, for the Bonferroni-corrected alpha. Set
    this to whatever your actual screening universe was (e.g. 1275 for
    all bivariate pairs across 51 instruments -- adjust if your 8
    candidates came from a different screen, e.g. triplets).

USAGE
-----
    import numpy as np
    from adf_train_holdout_check import run_checks

    prcSoFar = ...  # shape (51, n_days), n_days >= TRAIN_DAYS+HOLDOUT_DAYS
    run_checks(prcSoFar)
"""

import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# ============================================================
# EDIT THIS SECTION
# ============================================================

TICKER_TO_IDX = {
    "HETT": 7,
    "ULXY": 40,
    "OTCS": 6,
    "SMAH": 10,
    "ILVX": 46,
    "HUXZ": 8,
    "RTTH": 18,
"CTGI": 25,
"SRTX": 28,
"NAYO": 35,
"FWWG": 36,
"EELT": 37,
"BLBT": 41,
"HRTK": 44,
}
CANDIDATES = [
    ["HETT", "ULXY"],
    ["OTCS", "SMAH", "ILVX"],

    ["CTGI", "SRTX"],
    ["CTGI", "NAYO"],
    ["CTGI", "FWWG"],
    ["CTGI", "EELT"],
    ["CTGI", "BLBT"],
    ["CTGI", "HRTK"],

    ["SRTX", "NAYO"],
    ["SRTX", "FWWG"],
    ["SRTX", "EELT"],
    ["SRTX", "BLBT"],
    ["SRTX", "HRTK"],

    ["NAYO", "FWWG"],
    ["NAYO", "EELT"],
    ["NAYO", "BLBT"],
    ["NAYO", "HRTK"],

    ["FWWG", "EELT"],
    ["FWWG", "BLBT"],
    ["FWWG", "HRTK"],

    ["EELT", "BLBT"],
    ["EELT", "HRTK"],

    ["BLBT", "HRTK"]
]


TRAIN_DAYS = 335
HOLDOUT_DAYS = 165

ALPHA = 0.05
N_TOTAL_TESTS = 1275  # set to your actual screening universe size

# ============================================================


def _fit_johansen_beta(prices):
    """prices: shape (k, n_days). Returns the normalized cointegrating
    vector (first coefficient = 1) from the top Johansen eigenvector,
    fit on exactly this slice of data."""
    k, n = prices.shape
    data = prices.T  # statsmodels wants (n_obs, k_vars)
    jres = coint_johansen(data, det_order=0, k_ar_diff=1)
    top_evec = jres.evec[:, 0]  # eigenvector for the largest eigenvalue
    beta = top_evec / top_evec[0]  # normalize so first coeff = 1
    return beta


def _spread(prices, beta):
    return (beta[:, None] * prices).sum(axis=0)


def _half_life(spread):
    x, y = spread[:-1], spread[1:]
    if np.std(x) == 0:
        return float("nan")
    b, a = np.polyfit(x, y, 1)
    if not (0 < b < 1):
        return float("nan")
    return -np.log(2) / np.log(b)


def _adf(spread):
    stat, pval, *_ = adfuller(spread, autolag="AIC")
    return stat, pval


def run_checks(prcSoFar):
    prcSoFar = np.asarray(prcSoFar, dtype=float)
    n_instruments, n_days = prcSoFar.shape
    needed = TRAIN_DAYS + HOLDOUT_DAYS
    if n_days < needed:
        print(f"Need at least {needed} days (train {TRAIN_DAYS} + holdout "
              f"{HOLDOUT_DAYS}), only have {n_days}. Aborting.")
        return

    alpha_corrected = ALPHA / N_TOTAL_TESTS
    print(f"{'=' * 100}")
    print(f"Train = days [0:{TRAIN_DAYS}), Holdout = days [{TRAIN_DAYS}:{TRAIN_DAYS + HOLDOUT_DAYS})")
    print(f"alpha = {ALPHA}, Bonferroni-corrected alpha (N={N_TOTAL_TESTS}) = {alpha_corrected:.3e}")
    print(f"{'=' * 100}\n")

    rows = []
    for group in CANDIDATES:
        try:
            idxs = [TICKER_TO_IDX[t] for t in group]
        except KeyError as e:
            print(f"Skipping {group}: ticker {e} not in TICKER_TO_IDX.")
            continue

        prices_all = prcSoFar[idxs, :]
        prices_train = prices_all[:, :TRAIN_DAYS]
        prices_holdout = prices_all[:, TRAIN_DAYS:TRAIN_DAYS + HOLDOUT_DAYS]
        prices_pooled = prices_all[:, :TRAIN_DAYS + HOLDOUT_DAYS]

        # 1. Fit beta on TRAIN ONLY
        beta_train = _fit_johansen_beta(prices_train)
        spread_train = _spread(prices_train, beta_train)
        train_stat, train_pval = _adf(spread_train)
        hl_train = _half_life(spread_train)

        # 2. Apply TRAIN beta (no refit) to HOLDOUT, test independently
        spread_holdout = _spread(prices_holdout, beta_train)
        holdout_stat, holdout_pval = _adf(spread_holdout)
        hl_holdout = _half_life(spread_holdout)

        # 3. Reference only: fit + test on pooled train+holdout together
        beta_pooled = _fit_johansen_beta(prices_pooled)
        spread_pooled = _spread(prices_pooled, beta_pooled)
        pooled_stat, pooled_pval = _adf(spread_pooled)

        train_pass_raw = train_pval < ALPHA
        holdout_pass_raw = holdout_pval < ALPHA
        train_pass_bonf = train_pval < alpha_corrected
        holdout_pass_bonf = holdout_pval < alpha_corrected
        both_independent_pass_raw = train_pass_raw and holdout_pass_raw
        both_independent_pass_bonf = train_pass_bonf and holdout_pass_bonf

        rows.append({
            "group": group,
            "beta_train": beta_train,
            "train_pval": train_pval,
            "holdout_pval": holdout_pval,
            "pooled_pval": pooled_pval,
            "hl_train": hl_train,
            "hl_holdout": hl_holdout,
            "both_independent_pass_raw": both_independent_pass_raw,
            "both_independent_pass_bonf": both_independent_pass_bonf,
        })

        name = "/".join(group)
        print(f"--- {name} ---")
        print(f"  beta (train-fit): {np.round(beta_train, 6).tolist()}")
        print(f"  TRAIN   ADF p-value = {train_pval:.5g}  "
              f"(pass@alpha={ALPHA}: {train_pass_raw}, pass@bonferroni: {train_pass_bonf})  "
              f"half-life={hl_train:.1f}d")
        print(f"  HOLDOUT ADF p-value = {holdout_pval:.5g}  "
              f"(pass@alpha={ALPHA}: {holdout_pass_raw}, pass@bonferroni: {holdout_pass_bonf})  "
              f"half-life={hl_holdout:.1f}d  [using TRAIN-fitted beta, not refit]")
        print(f"  POOLED  ADF p-value = {pooled_pval:.5g}  (fit+test on full train+holdout -- reference only)")
        verdict = "PASS (independent, both splits)" if both_independent_pass_raw else "FAIL"
        verdict_bonf = "PASS (independent, both splits, Bonferroni)" if both_independent_pass_bonf else "FAIL (Bonferroni)"
        print(f"  => {verdict}   |   {verdict_bonf}")
        if pooled_pval < ALPHA and not both_independent_pass_raw:
            print("  ** WARNING: pooled test passes but train+holdout independently do NOT both pass. **")
            print("  ** This is exactly the failure mode of testing once on the full window. **")
        print()

    print(f"{'=' * 100}")
    print("SUMMARY")
    print(f"{'=' * 100}")
    n_pass_raw = sum(r["both_independent_pass_raw"] for r in rows)
    n_pass_bonf = sum(r["both_independent_pass_bonf"] for r in rows)
    print(f"{n_pass_raw}/{len(rows)} candidates pass ADF independently on BOTH train and holdout at alpha={ALPHA}.")
    print(f"{n_pass_bonf}/{len(rows)} candidates pass independently on BOTH at Bonferroni-corrected alpha.")
    for r in rows:
        name = "/".join(r["group"])
        flag = "OK" if r["both_independent_pass_raw"] else "FAIL"
        flag_b = "OK" if r["both_independent_pass_bonf"] else "FAIL"
        print(f"  {name:20s} train_p={r['train_pval']:.4g}  holdout_p={r['holdout_pval']:.4g}  "
              f"pooled_p={r['pooled_pval']:.4g}  raw={flag}  bonferroni={flag_b}")

    return rows


if __name__ == "__main__":
    import pandas as pd
    prcSoFar = pd.read_csv("./prices.txt", sep=r"\s+", header=0, index_col=None).T
    run_checks(prcSoFar)

