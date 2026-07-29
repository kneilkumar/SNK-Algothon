"""Algothon 2026 combined strategy -- audited 1000-day revision.

Primary live books
------------------
1. Ten disjoint cointegrated pairs.
2. Five standalone time-series signals.
3. Six expanding Ridge next-day return models.
4. Seventeen one-day directed lead-lag signals.
5. NPCK residual mean reversion and an AETS hybrid signal.
6. A hybrid ALGO signal.
7. Five pair-leg capacity overlays that only use otherwise-unused capacity.
8. MHRM Ridge fallback while MHRM/EAFC is flat.

Important interpretation
------------------------
PAIR_EXIT_Z is intentionally zero. This means a pair signal is held until an
opposite entry signal appears; it is a switch-only rule rather than a normal
entry/exit-band rule. The earlier combined file also behaved this way, but its
comments described it as an exit threshold, which was misleading.

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
TICKER_TO_IDX = {ticker: i for i, ticker in enumerate(TICKERS)}

FAIL_SAFE = True

# ---------------------------------------------------------------------------
# 1) Pair book: all pairs are disjoint.
# ---------------------------------------------------------------------------
PAIRS = (
    (1, 20),   # AENO / NWIG
    (10, 46),  # SMAH / ILVX
    (8, 27),   # HUXZ / ACAC
    (13, 45),  # EORC / NGTE
    (49, 50),  # MHRM / EAFC
    (18, 35),  # RTTH / NAYO
    (36, 41),  # FWWG / BLBT
    (25, 37),  # CTGI / EELT
    (33, 42),  # MTNS / BENI
    (31, 43),  # ACIX / ITPA
)

HEDGE_WINDOW = 300
Z_WINDOW = 60
PAIR_ENTRY_Z = 1.25
PAIR_EXIT_Z = 0.0  # switch-only: hold until an opposite entry signal
PAIR_ALLOC_FRACTION = 1.0
MIN_HEDGE_OBS = 120

# ---------------------------------------------------------------------------
# 2) Standalone time-series rules.
# direction: +1 momentum, -1 reversal.
# ---------------------------------------------------------------------------
TS_SIGNALS = (
    (7, 5, +1),    # HETT: 5-day momentum
    (30, 1, +1),   # RCRI: 1-day momentum
    (40, 40, -1),  # ULXY: 40-day reversal
    (14, 2, -1),   # CUBO: 2-day reversal
    (15, 2, -1),   # HRET: 2-day reversal
)
TS_ALLOC_FRACTION = 1.0

# ---------------------------------------------------------------------------
# 3) Expanding Ridge next-day return models.
# Each tuple is: (target index, ridge alpha, absolute prediction threshold).
# All 51 current-day returns are features; the target is its next-day return.
# ---------------------------------------------------------------------------
RIDGE_SPECS = (
    (5, 1000.0, 0.00010),  # AMRP
    (17, 1.0, 0.00025),    # DIHO
    (28, 100.0, 0.00025),  # SRTX
    (12, 1.0, 0.00025),    # MSDP
    (29, 1000.0, 0.00050), # GARI
    (9, 1.0, 0.00010),     # DUCT
)
RIDGE_MIN_OBS = 200
RIDGE_ALLOC_FRACTION = 1.0

# ---------------------------------------------------------------------------
# 4) Directed one-day lead-lag rules.
# Tuple: (predictor, target, direction). The target follows direction times
# the sign of the predictor's latest return.
# ---------------------------------------------------------------------------
LEAD_LAG_SIGNALS = (
    (17, 38, +1),  # DIHO -> HRND
    (9, 24, -1),   # DUCT -> RRES
    (2, 26, -1),   # LSST -> ALUT
    (13, 32, +1),  # EORC -> CCNS
    (40, 47, +1),  # ULXY -> FCSG
    (33, 4, +1),   # MTNS -> ELLT
    (26, 48, -1),  # ALUT -> FARS
    (25, 23, +1),  # CTGI -> AGVF
    (43, 19, +1),  # ITPA -> SPLZ
    (32, 44, +1),  # CCNS -> HTRK
    (12, 3, -1),   # MSDP -> SRNA
    (33, 6, +1),   # MTNS -> OTCS
    (0, 21, +1),   # ALGO -> MMBT
    (37, 22, +1),  # EELT -> MDGI
    (33, 16, +1),  # MTNS -> ANSO
    (45, 34, +1),  # NGTE -> IHOZ
    (9, 2, +1),    # DUCT -> LSST
)
LEAD_LAG_ALLOC_FRACTION = 1.0

# ---------------------------------------------------------------------------
# 5) Residual/hybrid books.
# ---------------------------------------------------------------------------
NPCK_IDX = 11
NPCK_BETA_WINDOW = 300
NPCK_Z_WINDOW = 60
NPCK_ENTRY_Z = 1.25
NPCK_EXIT_Z = 0.25
NPCK_ALLOC_FRACTION = 1.0

AETS_IDX = 39
AETS_LEAD_IDX = 40  # ULXY
AETS_BETA_WINDOW = 300
AETS_Z_WINDOW = 60
AETS_EXTREME_Z = 1.0
AETS_ALLOC_FRACTION = 1.0

# ALGO hybrid: at a large ALGO/raw-basket residual, trade mean reversion;
# otherwise follow RCRI's latest return. This is a predictive hybrid signal,
# not a claim that the raw-price basket exactly replicates ALGO.
ALGO_BETA_WINDOW = 300
ALGO_Z_WINDOW = 60
ALGO_EXTREME_Z = 1.75
ALGO_ALLOC_FRACTION = 1.0

# ---------------------------------------------------------------------------
# 6) MHRM fallback while the MHRM/EAFC pair is flat.
# ---------------------------------------------------------------------------
MHRM_IDX = 49
EAFC_IDX = 50
MHRM_FEATURES = (40, 38, 47, 50, 31)  # ULXY, HRND, FCSG, EAFC, ACIX
MHRM_RIDGE_ALPHA = 10.0
MHRM_MIN_OBS = 200
MHRM_RIDGE_ALLOC_FRACTION = 0.60

# ---------------------------------------------------------------------------
# 7) Pair-leg capacity overlays.
# These never reverse or reduce a pair leg. They fill only spare capacity when
# the directed signal agrees with the existing pair-leg direction.
# ---------------------------------------------------------------------------
PAIR_CAPACITY_OVERLAYS = (
    (13, 45, +1),  # EORC -> NGTE
    (21, 27, +1),  # MMBT -> ACAC
    (3, 50, +1),   # SRNA -> EAFC
    (29, 41, +1),  # GARI -> BLBT
    (43, 37, +1),  # ITPA -> EELT
)

_pair_signal = {i: 0 for i in range(len(PAIRS))}
_npck_signal = 0
_last_good_position = np.zeros(N_INST, dtype=int)


def reset_state() -> None:
    global _pair_signal, _npck_signal, _last_good_position
    _pair_signal = {i: 0 for i in range(len(PAIRS))}
    _npck_signal = 0
    _last_good_position = np.zeros(N_INST, dtype=int)


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


def _latest_return(prices: np.ndarray, idx: int) -> float:
    if prices.shape[1] < 2:
        return 0.0
    return float(prices[idx, -1] / prices[idx, -2] - 1.0)


def _full_position(idx: int, signal: int, prices_today: np.ndarray, fraction: float) -> int:
    price = float(prices_today[idx])
    if signal == 0 or price <= 0:
        return 0
    shares = int(np.floor(fraction * POSITION_LIMITS[idx] / price))
    return int(signal * shares)


def _pair_snapshot(prices: np.ndarray, pair: tuple[int, int]):
    n_days = prices.shape[1]
    if n_days < max(MIN_HEDGE_OBS, Z_WINDOW + 2):
        return None, None
    ia, ib = pair
    pa = prices[ia]
    pb = prices[ib]
    w = min(HEDGE_WINDOW, n_days)
    beta, const = _ols_hedge(pa[-w:], pb[-w:])
    if beta is None or const is None or beta <= 0:
        return None, None
    wz = min(Z_WINDOW, n_days)
    spread = pa[-wz:] - beta * pb[-wz:] - const
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
            signal = 1
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
        out[ia] += signal * qa
        out[ib] += -signal * qb
    return out


def _timeseries_positions(prices: np.ndarray) -> np.ndarray:
    out = np.zeros(N_INST, dtype=int)
    today = prices[:, -1]
    n_days = prices.shape[1]
    for idx, lookback, direction in TS_SIGNALS:
        if n_days < lookback + 1:
            continue
        move = float(prices[idx, -1] / prices[idx, -1 - lookback] - 1.0)
        signal = 1 if move > 0 else (-1 if move < 0 else 0)
        signal *= int(direction)
        out[idx] = _full_position(idx, signal, today, TS_ALLOC_FRACTION)
    return out


def _ridge_positions(prices: np.ndarray) -> np.ndarray:
    out = np.zeros(N_INST, dtype=int)
    n_days = prices.shape[1]
    if n_days < RIDGE_MIN_OBS + 3:
        return out

    returns = prices[:, 1:] / prices[:, :-1] - 1.0
    X = returns[:, :-1].T
    current_x = returns[:, -1]
    if X.shape[0] < RIDGE_MIN_OBS:
        return out

    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0, ddof=0)
    x_std = np.where(x_std < 1e-10, 1.0, x_std)
    Xs = (X - x_mean) / x_std
    current_xs = (current_x - x_mean) / x_std
    gram = Xs.T @ Xs
    today = prices[:, -1]

    # Group targets by alpha so one matrix factorisation handles several RHS.
    for alpha in sorted({spec[1] for spec in RIDGE_SPECS}):
        group = [spec for spec in RIDGE_SPECS if spec[1] == alpha]
        indices = [spec[0] for spec in group]
        Y = returns[indices, 1:].T
        y_mean = Y.mean(axis=0)
        Yc = Y - y_mean
        rhs = Xs.T @ Yc
        try:
            coef = np.linalg.solve(gram + alpha * np.eye(Xs.shape[1]), rhs)
        except np.linalg.LinAlgError:
            continue
        predictions = y_mean + current_xs @ coef
        for (idx, _alpha, threshold), prediction in zip(group, predictions):
            prediction = float(prediction)
            if not np.isfinite(prediction) or abs(prediction) <= threshold:
                continue
            signal = 1 if prediction > 0 else -1
            out[idx] = _full_position(idx, signal, today, RIDGE_ALLOC_FRACTION)
    return out


def _lead_lag_positions(prices: np.ndarray) -> np.ndarray:
    out = np.zeros(N_INST, dtype=int)
    if prices.shape[1] < 2:
        return out
    today = prices[:, -1]
    latest_returns = prices[:, -1] / prices[:, -2] - 1.0
    for predictor, target, direction in LEAD_LAG_SIGNALS:
        value = float(direction * latest_returns[predictor])
        signal = 1 if value > 0 else (-1 if value < 0 else 0)
        out[target] = _full_position(target, signal, today, LEAD_LAG_ALLOC_FRACTION)
    return out


def _npck_position(prices: np.ndarray) -> np.ndarray:
    global _npck_signal
    out = np.zeros(N_INST, dtype=int)
    n_days = prices.shape[1]
    if n_days < NPCK_Z_WINDOW + 2:
        return out
    p = prices[NPCK_IDX]
    algo = prices[0]
    w = min(NPCK_BETA_WINDOW, n_days)
    beta, const = _ols_hedge(p[-w:], algo[-w:])
    if beta is None or const is None:
        return out
    wz = min(NPCK_Z_WINDOW, n_days)
    residual = p[-wz:] - beta * algo[-wz:] - const
    sigma = float(residual.std(ddof=0))
    if sigma < 1e-9 or not np.isfinite(sigma):
        return out
    z = float((residual[-1] - residual.mean()) / sigma)
    if z > NPCK_ENTRY_Z:
        signal = -1
    elif z < -NPCK_ENTRY_Z:
        signal = 1
    elif abs(z) < NPCK_EXIT_Z:
        signal = 0
    else:
        signal = _npck_signal
    _npck_signal = signal
    out[NPCK_IDX] = _full_position(
        NPCK_IDX, signal, prices[:, -1], NPCK_ALLOC_FRACTION
    )
    return out


def _aets_hybrid_position(prices: np.ndarray) -> np.ndarray:
    out = np.zeros(N_INST, dtype=int)
    n_days = prices.shape[1]
    if n_days < AETS_Z_WINDOW + 2:
        return out
    p = prices[AETS_IDX]
    algo = prices[0]
    w = min(AETS_BETA_WINDOW, n_days)
    beta, const = _ols_hedge(p[-w:], algo[-w:])
    if beta is None or const is None:
        return out
    wz = min(AETS_Z_WINDOW, n_days)
    residual = p[-wz:] - beta * algo[-wz:] - const
    sigma = float(residual.std(ddof=0))
    if sigma < 1e-9 or not np.isfinite(sigma):
        return out
    z = float((residual[-1] - residual.mean()) / sigma)
    lead_return = _latest_return(prices, AETS_LEAD_IDX)
    lead_signal = 1 if lead_return > 0 else (-1 if lead_return < 0 else 0)
    signal = (-1 if z > 0 else 1) if abs(z) > AETS_EXTREME_Z else lead_signal
    out[AETS_IDX] = _full_position(
        AETS_IDX, signal, prices[:, -1], AETS_ALLOC_FRACTION
    )
    return out


def _algo_hybrid_position(prices: np.ndarray) -> np.ndarray:
    out = np.zeros(N_INST, dtype=int)
    n_days = prices.shape[1]
    if n_days < ALGO_Z_WINDOW + 2:
        return out
    algo = prices[0]
    raw_basket = prices[1:].mean(axis=0)
    w = min(ALGO_BETA_WINDOW, n_days)
    beta, const = _ols_hedge(algo[-w:], raw_basket[-w:])
    if beta is None or const is None:
        return out
    wz = min(ALGO_Z_WINDOW, n_days)
    residual = algo[-wz:] - beta * raw_basket[-wz:] - const
    sigma = float(residual.std(ddof=0))
    if sigma < 1e-9 or not np.isfinite(sigma):
        return out
    z = float((residual[-1] - residual.mean()) / sigma)
    rcri_return = _latest_return(prices, 30)
    lead_signal = 1 if rcri_return > 0 else (-1 if rcri_return < 0 else 0)
    signal = (-1 if z > 0 else 1) if abs(z) > ALGO_EXTREME_Z else lead_signal
    out[0] = _full_position(0, signal, prices[:, -1], ALGO_ALLOC_FRACTION)
    return out


def _mhrm_fallback_position(prices: np.ndarray, pair_positions: np.ndarray) -> np.ndarray:
    out = np.zeros(N_INST, dtype=int)
    if pair_positions[MHRM_IDX] != 0 or pair_positions[EAFC_IDX] != 0:
        return out
    if prices.shape[1] < MHRM_MIN_OBS + 3:
        return out
    returns = prices[:, 1:] / prices[:, :-1] - 1.0
    feature_idx = np.asarray(MHRM_FEATURES, dtype=int)
    X = returns[feature_idx, :-1].T
    y = returns[MHRM_IDX, 1:]
    if X.shape[0] < MHRM_MIN_OBS:
        return out
    xm = X.mean(axis=0)
    xs = X.std(axis=0, ddof=0)
    xs = np.where(xs < 1e-10, 1.0, xs)
    Xs = (X - xm) / xs
    ym = float(y.mean())
    try:
        coef = np.linalg.solve(
            Xs.T @ Xs + MHRM_RIDGE_ALPHA * np.eye(Xs.shape[1]),
            Xs.T @ (y - ym),
        )
    except np.linalg.LinAlgError:
        return out
    prediction = float(ym + ((returns[feature_idx, -1] - xm) / xs) @ coef)
    if not np.isfinite(prediction) or prediction == 0:
        return out
    signal = 1 if prediction > 0 else -1
    out[MHRM_IDX] = _full_position(
        MHRM_IDX, signal, prices[:, -1], MHRM_RIDGE_ALLOC_FRACTION
    )
    return out


def _apply_pair_capacity_overlays(positions: np.ndarray, prices: np.ndarray) -> None:
    if prices.shape[1] < 2:
        return
    latest_returns = prices[:, -1] / prices[:, -2] - 1.0
    today = prices[:, -1]
    for predictor, target, direction in PAIR_CAPACITY_OVERLAYS:
        value = float(direction * latest_returns[predictor])
        signal = 1 if value > 0 else (-1 if value < 0 else 0)
        if signal == 0:
            continue
        existing = int(positions[target])
        if existing != 0 and int(np.sign(existing)) != signal:
            continue
        price = float(today[target])
        max_shares = int(np.floor(POSITION_LIMITS[target] / price))
        spare = max(0, max_shares - abs(existing))
        positions[target] += signal * spare


def _run_strategy(prices: np.ndarray) -> np.ndarray:
    pair_positions = _pair_positions(prices)
    positions = pair_positions.copy()
    positions += _timeseries_positions(prices)
    positions += _ridge_positions(prices)
    positions += _lead_lag_positions(prices)
    positions += _npck_position(prices)
    positions += _aets_hybrid_position(prices)
    positions += _algo_hybrid_position(prices)
    positions += _mhrm_fallback_position(prices, pair_positions)
    _apply_pair_capacity_overlays(positions, prices)
    return positions


def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    global _last_good_position

    def _compute() -> np.ndarray:
        prices = np.asarray(prcSoFar, dtype=float)
        if prices.ndim != 2 or prices.shape[0] != N_INST or prices.shape[1] < 1:
            raise ValueError(f"Expected (51, n_days), got {prices.shape}")
        if not np.all(np.isfinite(prices[:, -1])) or np.any(prices[:, -1] <= 0):
            raise ValueError("Latest prices contain non-finite/non-positive values")
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
