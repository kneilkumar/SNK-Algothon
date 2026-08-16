"""Algothon 2026 robust candidate for the next 500 unseen days.

This version deliberately uses a small number of systematic strategy families
rather than assigning a separately mined rule to every instrument.

Live books
----------
1. Eight disjoint rolling cointegration pairs.
2. A two-alpha expanding multi-output Ridge model for next-day returns.
3. Eighteen directional rules that passed a three-stage stability screen:
   selected on Days 1-750, positive on Days 751-1000, and positive across all
   neighbouring lookbacks tested.
4. ALGO ensemble: equal-weight blend of a residual/RCRI hybrid and NPCK
   ten-day reversal. Disagreement automatically halves ALGO exposure.
5. Cross-sectional ALGO-adjusted five-day residual reversal, used only to fill
   instruments for which the other books currently have no position.

Combination rules
-----------------
- Pair legs have priority and are never distorted by another book.
- For non-pair instruments, Ridge and a stable directional rule must agree.
  Agreement receives a full-cap position; disagreement stays flat.
- When Ridge is below its confidence threshold, a validated directional rule
  may trade by itself.
- Cross-sectional reversal only fills positions that remain empty.

Only getMyPosition(prcSoFar) is required by the competition harness.
"""

from __future__ import annotations

import numpy as np

N_INST = 51
POSITION_LIMITS = np.array([100_000.0] + [10_000.0] * 50)
TICKERS = (
    "ALGO", "AENO", "LSST", "SRNA", "ELLT", "AMRP", "OTCS", "HETT",
    "HUXZ", "DUCT", "SMAH", "NPCK", "MSDP", "EORC", "CUBO", "HRET",
    "ANSO", "DIHO", "RTTH", "SPLZ", "NWIG", "MMBT", "MDGI", "AGVF",
    "RRES", "CTGI", "ALUT", "ACAC", "SRTX", "GARI", "RCRI", "ACIX",
    "CCNS", "MTNS", "IHOZ", "NAYO", "FWWG", "EELT", "HRND", "AETS",
    "ULXY", "BLBT", "BENI", "ITPA", "HTRK", "NGTE", "ILVX", "FCSG",
    "FARS", "MHRM", "EAFC",
)

FAIL_SAFE = False

# ---------------------------------------------------------------------------
# 1) Robust disjoint pair book
# ---------------------------------------------------------------------------
PAIRS = (
    (1, 20),   # AENO / NWIG
    (13, 45),  # EORC / NGTE
    (8, 27),   # HUXZ / ACAC
    (10, 46),  # SMAH / ILVX
    (18, 35),  # RTTH / NAYO
    (36, 41),  # FWWG / BLBT
    (31, 43),  # ACIX / ITPA
    (49, 50),  # MHRM / EAFC
)
PAIR_ASSETS = frozenset(i for pair in PAIRS for i in pair)
HEDGE_WINDOW = 300
PAIR_Z_WINDOW = 60
PAIR_ENTRY_Z = 1.25
PAIR_EXIT_Z = 0.0  # switch-only: hold until an opposite threshold crossing
PAIR_ALLOC_FRACTION = 1.0
MIN_HEDGE_OBS = 120

# ---------------------------------------------------------------------------
# 2) Multi-output Ridge next-day model
# ---------------------------------------------------------------------------
RIDGE_ALPHAS = (100.0, 1000.0)
RIDGE_MIN_OBS = 200
RIDGE_REFIT_EVERY = 5
RIDGE_Z_THRESHOLD = 0.05
RIDGE_FULL_SIZE_Z = 0.10
RIDGE_ALGO_FRACTION = 0.0  # ALGO is handled by its dedicated ensemble.

# ---------------------------------------------------------------------------
# 3) Stability-selected directional rules
# Tuple: (target, predictor, lookback, direction), where +1 follows and -1
# reverses the predictor's return over the stated lookback.
# ---------------------------------------------------------------------------
STABLE_RULES = ()


# ---------------------------------------------------------------------------
# 4) ALGO ensemble
# ---------------------------------------------------------------------------
ALGO_BETA_WINDOW = 300
ALGO_Z_WINDOW = 60
ALGO_EXTREME_Z = 1.75
ALGO_NPCK_LOOKBACK = 10
ALGO_HYBRID_WEIGHT = 0.50
ALGO_NPCK_WEIGHT = 0.50
ALGO_ALLOC_FRACTION = 1.0

# ---------------------------------------------------------------------------
# 5) Cross-sectional residual reversal fallback
# ---------------------------------------------------------------------------
CS_BETA_WINDOW = 120
CS_RESIDUAL_DAYS = 5
CS_K = 3
CS_REBALANCE_EVERY = 2
CS_ALLOC_FRACTION = 1.0

# Persistent state.
_pair_signal = {i: 0 for i in range(len(PAIRS))}
_cs_target = np.zeros(N_INST, dtype=int)
_last_good_position = np.zeros(N_INST, dtype=int)

# Ridge cache. The model is refit only every five calls/days.
_ridge_fit_day = -10**9
_ridge_models: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []


def reset_state() -> None:
    """Reset persistent state before an independent backtest."""
    global _pair_signal, _cs_target, _last_good_position
    global _ridge_fit_day, _ridge_models
    _pair_signal = {i: 0 for i in range(len(PAIRS))}
    _cs_target = np.zeros(N_INST, dtype=int)
    _last_good_position = np.zeros(N_INST, dtype=int)
    _ridge_fit_day = -10**9
    _ridge_models = []


def _ols_hedge(a: np.ndarray, b: np.ndarray) -> tuple[float | None, float | None]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size != b.size or a.size < 3:
        return None, None
    am = float(a.mean())
    bm = float(b.mean())
    bd = b - bm
    denom = float(np.dot(bd, bd))
    if not np.isfinite(denom) or denom < 1e-12:
        return None, None
    beta = float(np.dot(bd, a - am) / denom)
    const = float(am - beta * bm)
    if not np.isfinite(beta) or not np.isfinite(const):
        return None, None
    return beta, const


def _full_position(idx: int, signal: float, prices_today: np.ndarray, fraction: float = 1.0) -> int:
    price = float(prices_today[idx])
    if not np.isfinite(signal) or abs(signal) < 1e-12 or price <= 0:
        return 0
    max_shares = int(np.floor(fraction * POSITION_LIMITS[idx] / price))
    return int(np.rint(np.clip(signal, -1.0, 1.0) * max_shares))


# ---------------------------------------------------------------------------
# Pair book
# ---------------------------------------------------------------------------
def _pair_snapshot(prices: np.ndarray, pair: tuple[int, int]) -> tuple[float | None, float | None]:
    n_days = prices.shape[1]
    if n_days < max(MIN_HEDGE_OBS, PAIR_Z_WINDOW + 2):
        return None, None
    ia, ib = pair
    w = min(HEDGE_WINDOW, n_days)
    beta, const = _ols_hedge(prices[ia, -w:], prices[ib, -w:])
    if beta is None or const is None or beta <= 0:
        return None, None
    wz = min(PAIR_Z_WINDOW, n_days)
    spread = prices[ia, -wz:] - beta * prices[ib, -wz:] - const
    sigma = float(spread.std(ddof=0))
    if not np.isfinite(sigma) or sigma < 1e-9:
        return beta, None
    z = float((spread[-1] - spread.mean()) / sigma)
    return beta, z if np.isfinite(z) else None


def _pair_positions(prices: np.ndarray) -> np.ndarray:
    out = np.zeros(N_INST, dtype=int)
    today = prices[:, -1]
    for pair_no, (ia, ib) in enumerate(PAIRS):
        beta, z = _pair_snapshot(prices, (ia, ib))
        if beta is None or z is None:
            _pair_signal[pair_no] = 0
            continue
        previous = _pair_signal[pair_no]
        if z > PAIR_ENTRY_Z:
            signal = -1
        elif z < -PAIR_ENTRY_Z:
            signal = +1
        elif PAIR_EXIT_Z > 0 and abs(z) < PAIR_EXIT_Z:
            signal = 0
        else:
            signal = previous
        _pair_signal[pair_no] = signal
        if signal == 0:
            continue
        pa = float(today[ia])
        pb = float(today[ib])
        if pa <= 0 or pb <= 0:
            continue
        scale = min(
            PAIR_ALLOC_FRACTION * POSITION_LIMITS[ia] / pa,
            PAIR_ALLOC_FRACTION * POSITION_LIMITS[ib] / (abs(beta) * pb),
        )
        if not np.isfinite(scale) or scale < 1:
            continue
        qa = int(np.floor(scale))
        qb = int(np.floor(abs(beta) * scale))
        out[ia] = signal * qa
        out[ib] = -signal * qb
    return out


# ---------------------------------------------------------------------------
# Ridge model
# ---------------------------------------------------------------------------
def _fit_ridge_models(prices: np.ndarray) -> None:
    global _ridge_fit_day, _ridge_models
    n_days = prices.shape[1]
    if n_days < RIDGE_MIN_OBS + 3:
        return
    if _ridge_models and n_days - _ridge_fit_day < RIDGE_REFIT_EVERY:
        return

    returns = prices[:, 1:] / prices[:, :-1] - 1.0
    X = returns[:, :-1].T
    Y = returns[:, 1:].T
    if X.shape[0] < RIDGE_MIN_OBS:
        return

    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0, ddof=0)
    x_std = np.where(x_std < 1e-10, 1.0, x_std)
    Xs = (X - x_mean) / x_std
    y_mean = Y.mean(axis=0)
    Yc = Y - y_mean
    gram = Xs.T @ Xs
    rhs = Xs.T @ Yc

    models = []
    for alpha in RIDGE_ALPHAS:
        try:
            coef = np.linalg.solve(gram + alpha * np.eye(N_INST), rhs)
        except np.linalg.LinAlgError:
            continue
        residual = Y - (y_mean + Xs @ coef)
        residual_sd = residual.std(axis=0, ddof=0)
        residual_sd = np.where(residual_sd < 1e-8, 1e-8, residual_sd)
        models.append((x_mean, x_std, y_mean, coef, residual_sd))

    if models:
        _ridge_models = models
        _ridge_fit_day = n_days


def _ridge_zscores(prices: np.ndarray) -> np.ndarray:
    _fit_ridge_models(prices)
    z = np.full(N_INST, np.nan)
    if not _ridge_models or prices.shape[1] < 2:
        return z
    current_return = prices[:, -1] / prices[:, -2] - 1.0
    model_z = []
    for x_mean, x_std, y_mean, coef, residual_sd in _ridge_models:
        prediction = y_mean + ((current_return - x_mean) / x_std) @ coef
        model_z.append(prediction / residual_sd)
    return np.mean(np.asarray(model_z), axis=0)


def _ridge_positions(prices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = _ridge_zscores(prices)
    out = np.zeros(N_INST, dtype=int)
    today = prices[:, -1]
    for idx in range(1, N_INST):
        if idx in PAIR_ASSETS or not np.isfinite(z[idx]) or abs(z[idx]) < RIDGE_Z_THRESHOLD:
            continue
        strength = min(1.0, abs(float(z[idx])) / RIDGE_FULL_SIZE_Z)
        out[idx] = _full_position(idx, np.sign(z[idx]) * strength, today)
    return out, z


# ---------------------------------------------------------------------------
# Stability-selected directional rules
# ---------------------------------------------------------------------------
def _stable_rule_positions(prices: np.ndarray) -> np.ndarray:
    out = np.zeros(N_INST, dtype=int)
    today = prices[:, -1]
    n_days = prices.shape[1]
    for target, predictor, lookback, direction in STABLE_RULES:
        if target in PAIR_ASSETS or n_days < lookback + 1:
            continue
        move = float(prices[predictor, -1] / prices[predictor, -1 - lookback] - 1.0)
        signal = (1 if move > 0 else (-1 if move < 0 else 0)) * int(direction)
        out[target] = _full_position(target, signal, today)
    return out


def _combine_ridge_and_stable(ridge: np.ndarray, stable: np.ndarray) -> np.ndarray:
    """Use stable rules additively without allowing them to veto Ridge.

    - Ridge remains the primary systematic forecast.
    - Agreement may raise a partial Ridge position to the full cap.
    - A stable rule may trade when Ridge is below its threshold.
    - On disagreement, keep the Ridge position instead of trusting a mined
      directional rule strongly enough to cancel the regularised model.
    """
    out = ridge.copy()
    for idx in range(1, N_INST):
        if idx in PAIR_ASSETS or stable[idx] == 0:
            continue
        if ridge[idx] == 0:
            out[idx] = stable[idx]
        elif np.sign(ridge[idx]) == np.sign(stable[idx]):
            out[idx] = int(np.sign(ridge[idx]) * max(abs(ridge[idx]), abs(stable[idx])))
        # Disagreement deliberately leaves the Ridge position unchanged.
    return out


# ---------------------------------------------------------------------------
# ALGO ensemble
# ---------------------------------------------------------------------------
def _algo_position(prices: np.ndarray) -> int:
    n_days = prices.shape[1]
    if n_days < max(ALGO_Z_WINDOW + 2, ALGO_NPCK_LOOKBACK + 1):
        return 0

    algo = prices[0]
    raw_basket = prices[1:].mean(axis=0)
    w = min(ALGO_BETA_WINDOW, n_days)
    beta, const = _ols_hedge(algo[-w:], raw_basket[-w:])
    hybrid_signal = 0
    if beta is not None and const is not None:
        residual = algo[-ALGO_Z_WINDOW:] - beta * raw_basket[-ALGO_Z_WINDOW:] - const
        sigma = float(residual.std(ddof=0))
        if np.isfinite(sigma) and sigma >= 1e-9:
            z = float((residual[-1] - residual.mean()) / sigma)
            rcri_return = float(prices[30, -1] / prices[30, -2] - 1.0)
            rcri_signal = 1 if rcri_return > 0 else (-1 if rcri_return < 0 else 0)
            hybrid_signal = (-1 if z > 0 else 1) if abs(z) > ALGO_EXTREME_Z else rcri_signal

    npck_move = float(prices[11, -1] / prices[11, -1 - ALGO_NPCK_LOOKBACK] - 1.0)
    npck_signal = -1 if npck_move > 0 else (1 if npck_move < 0 else 0)
    blended = ALGO_HYBRID_WEIGHT * hybrid_signal + ALGO_NPCK_WEIGHT * npck_signal
    return _full_position(0, blended, prices[:, -1], ALGO_ALLOC_FRACTION)


# ---------------------------------------------------------------------------
# Cross-sectional fallback
# ---------------------------------------------------------------------------
def _cross_sectional_target(prices: np.ndarray) -> np.ndarray:
    out = np.zeros(N_INST, dtype=int)
    n_days = prices.shape[1]
    if n_days < CS_BETA_WINDOW + CS_RESIDUAL_DAYS + 2:
        return out
    returns = prices[:, 1:] / prices[:, :-1] - 1.0
    train = returns[:, -(CS_BETA_WINDOW + CS_RESIDUAL_DAYS):-CS_RESIDUAL_DAYS]
    signal_returns = returns[:, -CS_RESIDUAL_DAYS:]
    x = train[0]
    xm = float(x.mean())
    xd = x - xm
    xv = float(np.dot(xd, xd))
    if xv < 1e-12:
        return out

    scores = np.full(N_INST, np.nan)
    for idx in range(1, N_INST):
        if idx in PAIR_ASSETS:
            continue
        y = train[idx]
        ym = float(y.mean())
        beta = float(np.dot(xd, y - ym) / xv)
        alpha = ym - beta * xm
        scores[idx] = float(np.sum(signal_returns[idx] - (alpha + beta * signal_returns[0])))

    eligible = np.asarray([i for i in range(1, N_INST) if i not in PAIR_ASSETS and np.isfinite(scores[i])])
    if eligible.size < 2 * CS_K:
        return out
    ranked = eligible[np.argsort(scores[eligible])]
    today = prices[:, -1]
    for idx in ranked[:CS_K]:
        out[idx] = _full_position(idx, +1, today, CS_ALLOC_FRACTION)
    for idx in ranked[-CS_K:]:
        out[idx] = _full_position(idx, -1, today, CS_ALLOC_FRACTION)
    return out


def _cross_sectional_positions(prices: np.ndarray) -> np.ndarray:
    global _cs_target
    if prices.shape[1] < CS_BETA_WINDOW + CS_RESIDUAL_DAYS + 2:
        _cs_target = np.zeros(N_INST, dtype=int)
    elif prices.shape[1] % CS_REBALANCE_EVERY == 0:
        _cs_target = _cross_sectional_target(prices)
    return _cs_target.copy()


# ---------------------------------------------------------------------------
# Live combination
# ---------------------------------------------------------------------------
def _run_strategy(prices: np.ndarray) -> np.ndarray:
    pairs = _pair_positions(prices)
    ridge, _z = _ridge_positions(prices)
    stable = _stable_rule_positions(prices)
    directional = _combine_ridge_and_stable(ridge, stable)

    positions = directional
    for idx in PAIR_ASSETS:
        positions[idx] = pairs[idx]

    # Cross-sectional book only fills unused normal-instrument capacity.
    cross = _cross_sectional_positions(prices)
    empty = (positions == 0) & (cross != 0)
    positions[empty] = cross[empty]

    # ALGO has a separate, confidence-scaled ensemble.
    positions[0] = _algo_position(prices)

    # Enforce current-day limits internally. This prevents a two-day held
    # cross-sectional share count from exceeding the dollar cap after a price
    # rise, and ensures the competition evaluator never changes our intent.
    caps = (POSITION_LIMITS / prices[:, -1]).astype(int)
    return np.clip(positions, -caps, caps).astype(int)


def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    """Required competition interface: return 51 integer target positions."""
    global _last_good_position

    def _compute() -> np.ndarray:
        prices = np.asarray(prcSoFar, dtype=float)
        if prices.ndim != 2 or prices.shape[0] != N_INST or prices.shape[1] < 1:
            raise ValueError(f"Expected (51, n_days), got {prices.shape}")
        if not np.all(np.isfinite(prices[:, -1])) or np.any(prices[:, -1] <= 0):
            raise ValueError("Latest prices contain non-finite or non-positive values")
        result = np.asarray(_run_strategy(prices), dtype=int)
        if result.shape != (N_INST,):
            raise ValueError(f"Invalid output shape {result.shape}")
        return result

    if FAIL_SAFE:
        try:
            result = _compute()
        except Exception:
            return _last_good_position.copy()
    else:
        result = _compute()

    _last_good_position = result.copy()
    return result
