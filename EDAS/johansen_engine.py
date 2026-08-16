"""
johansen_engine.py

Systematic Johansen cointegration screening across all N-instrument
combinations (pairs, triplets, quadruplets, ...), with:

  * Bonferroni family-wise-error control across all C(nInst, group_size)
    tests, computed BEFORE looking at any results
  * standard statsmodels critical values (90/95/99%) reported alongside
    the adjusted threshold, so you can see what only clears the raw bar
  * mandatory post-screen validation: AR(1) half-life on the
    cointegrating residual, with a "workable half-life" cycle check
  * mandatory holdout confirmation: chronological train/test split,
    independent re-fit of Johansen + half-life on the holdout slice only

Only pairs (or N-tuples) that survive train-side Bonferroni + half-life
AND holdout cointegration + half-life count as real candidates.

Requires: numpy, scipy, pandas, statsmodels
"""

from __future__ import annotations

import math
import warnings
from itertools import combinations
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.vector_ar.vecm import coint_johansen


# ---------------------------------------------------------------------------
# Null-distribution simulation
#
# statsmodels only tabulates asymptotic critical values at 90/95/99%.
# A Bonferroni-adjusted alpha across ~1000+ tests (e.g. 0.05/1275 ~ 3.9e-5)
# falls far outside that table, so we simulate the null distribution of the
# trace statistic directly (independent random walks -> no cointegration by
# construction) at the actual group size and training sample length, and
# read the adjusted critical value off that simulated distribution.
#
# For alpha this small, even a few thousand simulations won't have enough
# raw observations that far into the tail, so beyond a resolution cutoff we
# fit a Generalized Pareto tail model (peaks-over-threshold) and extrapolate.
# This is a standard extreme-value approach, but it IS an approximation --
# treat it as directionally correct rather than exact, and raise n_null_sims
# if precision here matters a lot to you.
# ---------------------------------------------------------------------------

def simulate_null_trace_stats(
    group_size: int,
    n_obs: int,
    n_sims: int = 4000,
    det_order: int = 0,
    k_ar_diff: int = 1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Monte-Carlo the null distribution of the Johansen trace statistic
    (H0: rank <= 0) for `group_size` independent random walks of length
    `n_obs`. Depends only on group_size, n_obs, det_order, k_ar_diff --
    NOT on which instruments -- so it's simulated once and reused across
    every combination of that group size.
    """
    rng = np.random.default_rng(seed)
    out = np.full(n_sims, np.nan)
    for i in range(n_sims):
        walks = np.cumsum(rng.standard_normal((n_obs, group_size)), axis=0)
        try:
            res = coint_johansen(walks, det_order, k_ar_diff)
            out[i] = res.trace_stat[0]
        except Exception:
            continue
    out = out[~np.isnan(out)]
    if len(out) < 0.5 * n_sims:
        warnings.warn(
            f"Only {len(out)}/{n_sims} null simulations succeeded "
            "(numerical issues on degenerate random-walk draws are usually "
            "the cause). Consider raising n_null_sims."
        )
    return out


def _gpd_tail_fit(null_stats: np.ndarray, tail_quantile: float = 0.90):
    threshold = np.quantile(null_stats, tail_quantile)
    exceedances = null_stats[null_stats > threshold] - threshold
    if len(exceedances) < 30:
        return None
    c, loc, scale = stats.genpareto.fit(exceedances, floc=0)
    p_exceed = len(exceedances) / len(null_stats)
    return threshold, c, scale, p_exceed


def critical_value_from_null(null_stats: np.ndarray, alpha: float,
                              tail_quantile: float = 0.90) -> float:
    """(1 - alpha) critical value from the simulated null distribution.
    Empirical quantile when alpha is well within simulation resolution;
    GPD tail extrapolation otherwise (see module docstring caveat)."""
    n = len(null_stats)
    if alpha * n > 20:
        return float(np.quantile(null_stats, 1 - alpha))
    fit = _gpd_tail_fit(null_stats, tail_quantile)
    if fit is None:
        warnings.warn(
            "Not enough tail exceedances to fit a GPD model; falling back "
            "to the raw empirical quantile, which is likely "
            "under-conservative for this alpha. Raise n_null_sims."
        )
        return float(np.quantile(null_stats, 1 - alpha))
    threshold, c, scale, p_exceed = fit
    target_tail_p = alpha / p_exceed
    if target_tail_p >= 1:
        return float(threshold)
    z = stats.genpareto.ppf(1 - target_tail_p, c, loc=0, scale=scale)
    return float(threshold + z)


def null_p_value(null_stats: np.ndarray, observed_stat: float,
                  tail_quantile: float = 0.90) -> float:
    """Approximate p-value of an observed trace stat against the simulated
    null (empirical in the bulk, GPD-extrapolated beyond the simulated
    range)."""
    n = len(null_stats)
    if observed_stat <= np.max(null_stats):
        return float(np.mean(null_stats >= observed_stat))
    fit = _gpd_tail_fit(null_stats, tail_quantile)
    if fit is None:
        return float(np.mean(null_stats >= observed_stat))
    threshold, c, scale, p_exceed = fit
    z = observed_stat - threshold
    tail_p = 1 - stats.genpareto.cdf(z, c, loc=0, scale=scale)
    return float(p_exceed * tail_p)


# ---------------------------------------------------------------------------
# Per-combination Johansen fit + half-life
# ---------------------------------------------------------------------------

def _run_johansen(price_slice: np.ndarray, det_order: int, k_ar_diff: int) -> dict:
    """price_slice: (n_obs, group_size) level data. Returns the H0: rank<=0
    trace stat, tabulated critical values, and the cointegrating vector
    associated with the largest eigenvalue."""
    res = coint_johansen(price_slice, det_order, k_ar_diff)
    return {
        "trace_stat": float(res.trace_stat[0]),
        "crit_90": float(res.trace_stat_crit_vals[0, 0]),
        "crit_95": float(res.trace_stat_crit_vals[0, 1]),
        "crit_99": float(res.trace_stat_crit_vals[0, 2]),
        "max_eig_stat": float(res.max_eig_stat[0]),
        "eigenvalue": float(res.eig[0]),
        "evec": np.asarray(res.evec[:, 0], dtype=float),
    }


def _normalize_vector(evec: np.ndarray) -> np.ndarray:
    """Scale so the first coefficient is 1 (hedge-ratio form: P_0 = -sum of
    the rest). Falls back to a unit-norm vector if the first coefficient is
    ~0 (degenerate for hedge-ratio interpretation, but still a valid
    direction)."""
    if abs(evec[0]) > 1e-8:
        return evec / evec[0]
    return evec / np.linalg.norm(evec)


def _ar1_half_life(residual: np.ndarray):
    """Fit resid[t] = c + phi * resid[t-1] + eps by OLS. Returns
    (half_life, phi). half_life is np.inf if phi is outside (0, 1), i.e.
    not mean-reverting (unit root / explosive) or oscillatory."""
    y = residual[1:]
    x = residual[:-1]
    X = np.column_stack([np.ones(len(x)), x])
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    phi = float(coefs[1])
    if not (0 < phi < 1):
        return float("inf"), phi
    return float(np.log(0.5) / np.log(phi)), phi


def _cycles_ok(half_life: float, n_obs: int, min_cycles: float) -> bool:
    return np.isfinite(half_life) and half_life > 0 and (n_obs / half_life) >= min_cycles


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

def johansen_engine(
    prices: np.ndarray,
    instrument_names: Optional[Sequence[str]] = None,
    group_size: int = 2,
    train_frac: float = 0.67,
    fwer_alpha: float = 0.05,
    min_cycles: float = 3.5,
    det_order: int = 0,
    k_ar_diff: int = 1,
    n_null_sims: int = 4000,
    holdout_alpha_level: str = "95",
    max_tests: int = 50_000,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Screen all `group_size`-way combinations of instruments for Johansen
    cointegration, controlling FWER via Bonferroni, then validate survivors
    with an AR(1) half-life check and an independent holdout re-fit.

    Parameters
    ----------
    prices : array, shape (nInst, nt)
        Price LEVELS, training slice only (or your full history -- the
        function does the chronological train/holdout split itself, see
        `train_frac`).
    instrument_names : list of str, optional
        Length nInst. Defaults to inst_0, inst_1, ...
    group_size : int, default 2
        How many instruments per combination: 2 = pairs, 3 = triplets,
        4 = quadruplets, etc. This is the parameter you asked to make
        adjustable -- basket cointegration is just group_size > 2.
    train_frac : float, default 0.67
        Chronological split fraction. First `train_frac` of the columns
        are used for discovery/screening; the remainder is the untouched
        holdout used only for confirmation.
    fwer_alpha : float, default 0.05
        Family-wise error rate to control across all C(nInst, group_size)
        tests. The per-test Bonferroni threshold (fwer_alpha / n_tests) is
        computed once, up front, before any results are inspected.
    min_cycles : float, default 3.5
        Minimum number of AR(1) half-lives that must fit inside the window
        the half-life was estimated on, for it to count as tradeable.
    det_order : int, default 0
        Passed to statsmodels' coint_johansen: -1 no deterministic term,
        0 constant, 1 constant + trend. 0 is the usual choice for price
        levels that may have drift.
    k_ar_diff : int, default 1
        Number of lagged differences in the underlying VECM (i.e. a
        VAR(k_ar_diff + 1) in levels). Fixed for all combinations for
        speed; there's no per-combination lag selection here.
    n_null_sims : int, default 4000
        Monte Carlo draws used to build the null trace-stat distribution
        for the Bonferroni threshold. Higher = more reliable, slower.
        See module docstring for the tail-extrapolation caveat.
    holdout_alpha_level : {"90", "95", "99"}, default "95"
        Which tabulated statsmodels critical value to require on the
        (single, per-candidate) holdout re-fit. Bonferroni is not reapplied
        here since by this point you're only re-testing a short list, not
        1000+ combinations.
    max_tests : int, default 50_000
        Safety cap on C(nInst, group_size). Raises rather than silently
        grinding for hours -- pass a higher value explicitly if you really
        want to run a big combinatorial screen (e.g. group_size=4 on 51
        instruments is ~250k tests).
    seed : int, default 42
        RNG seed for the null-distribution simulation (reproducibility).
    verbose : bool, default True
        Print progress/summary lines.

    Returns
    -------
    dict with keys:
      'all_results'   : DataFrame, one row per combination tested
      'candidates'    : DataFrame, the final short list (survives
                        train-side Bonferroni + half-life AND holdout
                        cointegration + half-life)
      'meta'          : dict of run metadata (n_tests, thresholds, split
                        sizes, etc.)
    """
    prices = np.asarray(prices, dtype=float)
    n_inst, n_t = prices.shape
    if instrument_names is None:
        instrument_names = [f"inst_{i}" for i in range(n_inst)]
    assert len(instrument_names) == n_inst, "instrument_names length must match nInst"
    assert 2 <= group_size <= n_inst, "group_size must be between 2 and nInst"
    assert holdout_alpha_level in ("90", "95", "99")

    n_tests = math.comb(n_inst, group_size)
    if n_tests > max_tests:
        raise ValueError(
            f"C({n_inst},{group_size}) = {n_tests:,} combinations exceeds "
            f"max_tests={max_tests:,}. This will be slow, and the "
            f"Bonferroni threshold at that many tests is extremely strict "
            f"regardless. Pass a higher max_tests explicitly if you want "
            f"to proceed."
        )
    bonferroni_alpha = fwer_alpha / n_tests

    split = int(round(n_t * train_frac))
    if split < group_size + 5 or (n_t - split) < group_size + 5:
        raise ValueError(
            f"train_frac={train_frac} gives train={split}, holdout={n_t - split} "
            f"observations -- too short relative to group_size={group_size}."
        )
    train, holdout = prices[:, :split], prices[:, split:]

    if verbose:
        print(f"[johansen_engine] {n_inst} instruments, group_size={group_size} "
              f"-> {n_tests:,} combinations")
        print(f"[johansen_engine] train={train.shape[1]} obs, "
              f"holdout={holdout.shape[1]} obs (train_frac={train_frac})")
        print(f"[johansen_engine] FWER alpha={fwer_alpha}, "
              f"Bonferroni per-test alpha={bonferroni_alpha:.3e}")
        print(f"[johansen_engine] simulating null trace-stat distribution "
              f"({n_null_sims} sims, k={group_size}, T={train.shape[1]})...")

    null_stats = simulate_null_trace_stats(
        group_size, train.shape[1], n_sims=n_null_sims,
        det_order=det_order, k_ar_diff=k_ar_diff, seed=seed,
    )
    raw_crit_sim = critical_value_from_null(null_stats, fwer_alpha)
    bonf_crit_sim = critical_value_from_null(null_stats, bonferroni_alpha)
    if verbose:
        print(f"[johansen_engine] simulated raw crit (alpha={fwer_alpha}): "
              f"{raw_crit_sim:.2f}")
        print(f"[johansen_engine] simulated Bonferroni crit "
              f"(alpha={bonferroni_alpha:.2e}): {bonf_crit_sim:.2f}")

    rows = []
    fits_by_combo = {}
    for combo in combinations(range(n_inst), group_size):
        idx = list(combo)
        train_slice = train[idx, :].T
        try:
            r = _run_johansen(train_slice, det_order, k_ar_diff)
        except Exception:
            continue

        p_sim = null_p_value(null_stats, r["trace_stat"])
        row = {
            "combo": tuple(instrument_names[i] for i in idx),
            "combo_idx": combo,
            "trace_stat": r["trace_stat"],
            "crit_90_tab": r["crit_90"],
            "crit_95_tab": r["crit_95"],
            "crit_99_tab": r["crit_99"],
            "pass_90_tabulated": r["trace_stat"] > r["crit_90"],
            "pass_95_tabulated": r["trace_stat"] > r["crit_95"],
            "pass_99_tabulated": r["trace_stat"] > r["crit_99"],
            "p_value_sim": p_sim,
            "pass_raw_sim": r["trace_stat"] > raw_crit_sim,
            "pass_bonferroni": r["trace_stat"] > bonf_crit_sim,
            "coint_vector_normalized": _normalize_vector(r["evec"]),
        }
        rows.append(row)
        fits_by_combo[combo] = (r, idx, train_slice)

    if verbose:
        n_pass95 = sum(row["pass_95_tabulated"] for row in rows)
        n_pass_bonf = sum(row["pass_bonferroni"] for row in rows)
        print(f"[johansen_engine] {n_pass95}/{len(rows)} pass raw 95% (tabulated) "
              f"-- expect most 'passes' to live here, that's the noise floor")
        print(f"[johansen_engine] {n_pass_bonf}/{len(rows)} pass the "
              f"Bonferroni-adjusted threshold")

    # ---- post-screen validation: AR(1) half-life on TRAIN residual,
    #      for Bonferroni survivors only
    for row in rows:
        if not row["pass_bonferroni"]:
            row["half_life_train"] = np.nan
            row["phi_train"] = np.nan
            row["half_life_ok_train"] = False
            continue
        r, idx, train_slice = fits_by_combo[row["combo_idx"]]
        resid = train_slice @ r["evec"]
        hl, phi = _ar1_half_life(resid)
        row["half_life_train"] = hl
        row["phi_train"] = phi
        row["half_life_ok_train"] = _cycles_ok(hl, train_slice.shape[0], min_cycles)

    train_survivors = [row for row in rows
                       if row["pass_bonferroni"] and row["half_life_ok_train"]]
    if verbose:
        print(f"[johansen_engine] {len(train_survivors)} combo(s) survive "
              f"Bonferroni + train-side half-life screen -> holdout check")

    # ---- holdout confirmation: independent re-fit, standard tabulated
    #      critical value (not Bonferroni -- this is a short list by now)
    crit_key = f"crit_{holdout_alpha_level}"
    for row in rows:
        row["holdout_trace_stat"] = np.nan
        row["holdout_crit"] = np.nan
        row["holdout_pass"] = False
        row["holdout_coint_vector_normalized"] = None
        row["holdout_half_life"] = np.nan
        row["holdout_phi"] = np.nan
        row["holdout_half_life_ok"] = False
        row["final_candidate"] = False

    for row in train_survivors:
        idx = fits_by_combo[row["combo_idx"]][1]
        holdout_slice = holdout[idx, :].T
        try:
            rh = _run_johansen(holdout_slice, det_order, k_ar_diff)
        except Exception:
            continue

        holdout_pass = rh["trace_stat"] > rh[crit_key]
        resid_h = holdout_slice @ rh["evec"]
        hl_h, phi_h = _ar1_half_life(resid_h)
        hl_h_ok = _cycles_ok(hl_h, holdout_slice.shape[0], min_cycles)

        row["holdout_trace_stat"] = rh["trace_stat"]
        row["holdout_crit"] = rh[crit_key]
        row["holdout_pass"] = holdout_pass
        row["holdout_coint_vector_normalized"] = _normalize_vector(rh["evec"])
        row["holdout_half_life"] = hl_h
        row["holdout_phi"] = phi_h
        row["holdout_half_life_ok"] = hl_h_ok
        row["final_candidate"] = bool(holdout_pass and hl_h_ok)

    all_results = pd.DataFrame(rows).sort_values("trace_stat", ascending=False).reset_index(drop=True)
    candidates = all_results[all_results["final_candidate"]].reset_index(drop=True)

    if verbose:
        print(f"[johansen_engine] {len(candidates)} final candidate(s) "
              f"survive train screen + independent holdout confirmation")

    meta = {
        "n_inst": n_inst,
        "n_t": n_t,
        "group_size": group_size,
        "n_tests": n_tests,
        "train_obs": train.shape[1],
        "holdout_obs": holdout.shape[1],
        "fwer_alpha": fwer_alpha,
        "bonferroni_alpha": bonferroni_alpha,
        "raw_crit_sim": raw_crit_sim,
        "bonferroni_crit_sim": bonf_crit_sim,
        "min_cycles": min_cycles,
        "det_order": det_order,
        "k_ar_diff": k_ar_diff,
        "holdout_alpha_level": holdout_alpha_level,
    }

    return {"all_results": all_results, "candidates": candidates, "meta": meta}


# ---------------------------------------------------------------------------
# Convenience loader for the common "prices.txt" competition format
# (rows = time steps, columns = instruments, whitespace-delimited) ->
# returns the nInst x nt array this engine expects. Adjust if your file's
# layout differs.
# ---------------------------------------------------------------------------

def load_price_matrix(path: str) -> np.ndarray:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    return df.values.T  # nt x nInst -> nInst x nt
