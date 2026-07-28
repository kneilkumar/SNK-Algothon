"""Algothon 2026 live strategy -- 1000-day revision.

Live components
---------------
1) Six walk-forward validated cointegrated pairs.
2) Cross-sectional ALGO-adjusted residual-return reversal.
3) A small time-series momentum book on HETT and RCRI.
4) A Ridge-regression fallback on MHRM, used only while the MHRM/EAFC pair is flat.

MHRM conflict rule
------------------
MHRM/EAFC remains the primary use of MHRM.  The Ridge model is allowed to trade
MHRM only when that pair has no active position, so the regression signal never
distorts the pair hedge ratio or competes for the same instrument cap.

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

# ---------------------------------------------------------------------------
# Component toggles
# ---------------------------------------------------------------------------
ENABLE_PAIRS = True
ENABLE_CROSS_SECTIONAL = True
ENABLE_TIMESERIES = True
ENABLE_MHRM_RIDGE_FALLBACK = True

# Keep False during development so errors are visible.  Consider True only
# for a final submission after all local testing has passed.
FAIL_SAFE = False

# ---------------------------------------------------------------------------
# 1) Cointegrated pair book
# ---------------------------------------------------------------------------
# The original four pairs remained profitable on the new 250 days. MHRM/EAFC
# and RTTH/NAYO gained repeated positive walk-forward evidence on the 1000-day
# file. MHRM/EAFC replaces the weaker live Ridge signal on MHRM; RTTH/NAYO
# adds a sixth independent relative-value leg.
PAIRS = [
    (1, 20),   # AENO / NWIG
    (10, 46),  # SMAH / ILVX
    (8, 27),   # HUXZ / ACAC
    (13, 45),  # EORC / NGTE
    (49, 50),  # MHRM / EAFC
    (18, 35),  # RTTH / NAYO
]

# --- ADDITIONS (from independent cointegration scan; each ablation-tested) ---
# Selected on days [0,500), validated on [500,750). Only pairs that were
# profitable out-of-sample are listed. Toggled via ENABLE_EXTRA_PAIRS.
EXTRA_PAIRS = [
    (36, 41),  # FWWG / BLBT   oosSR 2.07
    (25, 37),  # CTGI / EELT   oosSR 1.93
    (7, 40),   # HETT / ULXY   oosSR 1.91
    (9, 20),   # DUCT / NWIG   oosSR 1.44
    (33, 42),  # MTNS / BENI   oosSR 1.26
    (31, 43),  # ACIX / ITPA   oosSR 1.09
]
ENABLE_EXTRA_PAIRS = True

# Residual mean-reversion books (price level netted against ALGO / basket).
# These only trade instruments NOT claimed by the pair or momentum books.
ALGO_RESID_MEANREV = [39, 15, 28, 11]   # AETS, HRET, SRTX, NPCK
BASKET_RESID_MEANREV = [16, 2, 34]   # ANSO, LSST, IHOZ
ENABLE_RESID_BOOKS = True
RESID_BETA_WINDOW = 300
RESID_Z_WINDOW = 60
RESID_ENTRY_Z = 1.25
RESID_EXIT_Z = 0.0
RESID_ALLOC_FRACTION = 0.99

# --- ALGO book -------------------------------------------------------------
# ALGO carries a $100k cap (10x every other name) and a 0.2bp commission
# (5x cheaper), so it is by far the largest single source of PnL if it can be
# traded at all. The pair-level Engle-Granger test on ALGO vs the basket is
# insignificant, but the z-score of the ALGO/basket RESIDUAL still mean-reverts:
# every configuration in this family was profitable on BOTH the validation
# window (500-750) and the untouched window (750-1000), which is the evidence
# that matters. Chosen config sits mid-family rather than at the optimum.
ENABLE_ALGO_BOOK = True
ALGO_BETA_WINDOW = 300
ALGO_Z_WINDOW = 60
ALGO_ENTRY_Z = 1.25
ALGO_EXIT_Z = 0.0
ALGO_ALLOC_FRACTION = 0.99

HEDGE_WINDOW = 300
Z_WINDOW = 60       # revised from 40 after 8-fold robustness testing
ENTRY_Z = 1.25
EXIT_Z = 0.0
PAIR_ALLOC_FRACTION = 0.99
MIN_HEDGE_OBS = 120

# ---------------------------------------------------------------------------
# 2) Cross-sectional residual-return reversal
# ---------------------------------------------------------------------------
# Fit each asset's return beta to ALGO using the 120 returns immediately
# before the 5-day signal window.  Rank the 5-day cumulative idiosyncratic
# residuals, buy the 3 weakest and short the 3 strongest.
#
# On the 1000-day file this configuration was profitable in every expanding
# validation fold. Daily rebalancing was more robust than the old 2-day rule
# once the final disjoint universe was enforced.
CS_BETA_WINDOW = 120
CS_RESIDUAL_DAYS = 5
CS_K = 6
CS_REBALANCE_EVERY = 1
CS_ALLOC_FRACTION = 0.99

# ---------------------------------------------------------------------------
# 3) Time-series momentum book
# ---------------------------------------------------------------------------
# A deliberately tiny, interpretable book selected from a pre-specified grid
# of lookbacks {1,2,3,5,10,20,40} and directions {momentum,reversal}.
# Both signals were profitable in all 8 chronological validation folds.
#
# HETT: follow the sign of the last 5-day return.
# RCRI: follow the sign of the last 1-day return.
TS_SIGNALS = (
    (7, 5, +1),   # HETT 5-day momentum
    (30, 1, +1),  # RCRI 1-day momentum
)
TS_ALLOC_FRACTION = 0.99

# ---------------------------------------------------------------------------
# 4) MHRM Ridge fallback
# ---------------------------------------------------------------------------
# Leakage-safe expanding Ridge model:
# current feature returns -> next-day MHRM return.  This component is only
# permitted to trade while the MHRM/EAFC pair is flat.
MHRM_IDX = 49
EAFC_IDX = 50
MHRM_FEATURES = (40, 38, 47, 50, 31)  # ULXY, HRND, FCSG, EAFC, ACIX
MHRM_RIDGE_ALPHA = 10.0
MHRM_MIN_OBS = 200
MHRM_RIDGE_ALLOC_FRACTION = 0.60
MHRM_PREDICTION_THRESHOLD = 0.0

# ---------------------------------------------------------------------------
# Persistent state
# ---------------------------------------------------------------------------
_pair_signal: dict[int, int] = {i: 0 for i in range(len(PAIRS) + len(EXTRA_PAIRS))}
_cs_target = np.zeros(N_INST, dtype=int)
_resid_signal = {i: 0 for i in (ALGO_RESID_MEANREV + BASKET_RESID_MEANREV)}
_algo_signal = 0
_last_good_position = np.zeros(N_INST, dtype=int)


def reset_state() -> None:
    """Reset persistent strategy state before an independent backtest."""
    global _pair_signal, _cs_target, _last_good_position, _resid_signal, _algo_signal
    _pair_signal = {i: 0 for i in range(len(PAIRS) + len(EXTRA_PAIRS))}
    _resid_signal = {i: 0 for i in (ALGO_RESID_MEANREV + BASKET_RESID_MEANREV)}
    _algo_signal = 0
    _cs_target = np.zeros(N_INST, dtype=int)
    _last_good_position = np.zeros(N_INST, dtype=int)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def _ols_hedge(a: np.ndarray, b: np.ndarray) -> tuple[float | None, float | None]:
    """Fit a = beta*b + const by closed-form OLS."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size != b.size or a.size < 3:
        return None, None

    a_mean = float(a.mean())
    b_mean = float(b.mean())
    b_dev = b - b_mean
    denom = float(np.dot(b_dev, b_dev))
    if not np.isfinite(denom) or denom < 1e-12:
        return None, None

    beta = float(np.dot(b_dev, a - a_mean) / denom)
    const = float(a_mean - beta * b_mean)
    if not np.isfinite(beta) or not np.isfinite(const):
        return None, None
    return beta, const


def _daily_returns(prices: np.ndarray) -> np.ndarray:
    return prices[:, 1:] / prices[:, :-1] - 1.0


def _active_pairs():
    """Base pairs plus (optionally) the independently-validated extra pairs."""
    return PAIRS + (EXTRA_PAIRS if ENABLE_EXTRA_PAIRS else [])


def _pair_assets() -> frozenset[int]:
    return frozenset(idx for pair in _active_pairs() for idx in pair)


def _resid_assets() -> frozenset[int]:
    if not ENABLE_RESID_BOOKS:
        return frozenset()
    claimed = frozenset(_pair_assets()) | _timeseries_assets()
    return frozenset(i for i in (ALGO_RESID_MEANREV + BASKET_RESID_MEANREV)
                     if i not in claimed)


def _active_ts_signals():
    """Momentum signals, minus any instrument now claimed by the pair book.
    Prevents HETT being traded by both the HETT/ULXY pair and the momentum book."""
    claimed = frozenset(idx for pair in _active_pairs() for idx in pair)
    return tuple(s for s in TS_SIGNALS if s[0] not in claimed)


def _timeseries_assets() -> frozenset[int]:
    return frozenset(idx for idx, _lookback, _direction in _active_ts_signals())


def _cross_excluded() -> frozenset[int]:
    """Names reserved by the other live books, so positions never conflict."""
    return frozenset({0, *_pair_assets(), *_timeseries_assets(), *_resid_assets()})


# ---------------------------------------------------------------------------
# Pair strategy
# ---------------------------------------------------------------------------
def _pair_snapshot(
    prc_so_far: np.ndarray,
    pair: tuple[int, int],
) -> tuple[float | None, float | None, float | None, float | None]:
    """Return (beta, const, z, spread_std) using information available today."""
    idx_a, idx_b = pair
    n_days = prc_so_far.shape[1]
    if n_days < max(MIN_HEDGE_OBS, Z_WINDOW + 2):
        return None, None, None, None

    pa = prc_so_far[idx_a]
    pb = prc_so_far[idx_b]
    w_hedge = min(HEDGE_WINDOW, n_days)
    beta, const = _ols_hedge(pa[-w_hedge:], pb[-w_hedge:])
    if beta is None or const is None or beta <= 0:
        return None, None, None, None

    w_z = min(Z_WINDOW, n_days)
    spread = pa[-w_z:] - beta * pb[-w_z:] - const
    sigma = float(spread.std(ddof=0))
    if not np.isfinite(sigma) or sigma < 1e-9:
        return beta, const, None, sigma

    z = float((spread[-1] - spread.mean()) / sigma)
    return beta, const, z if np.isfinite(z) else None, sigma


def _add_pair_position(
    positions: np.ndarray,
    prices_today: np.ndarray,
    idx_a: int,
    idx_b: int,
    beta: float,
    signal: int,
) -> None:
    if signal == 0:
        return

    price_a = float(prices_today[idx_a])
    price_b = float(prices_today[idx_b])
    if price_a <= 0 or price_b <= 0:
        return

    scale = min(
        PAIR_ALLOC_FRACTION * POSITION_LIMITS[idx_a] / price_a,
        PAIR_ALLOC_FRACTION * POSITION_LIMITS[idx_b] / (abs(beta) * price_b),
    )
    if not np.isfinite(scale) or scale < 1:
        return

    shares_a = int(np.floor(scale))
    shares_b = int(np.floor(abs(beta) * scale))
    positions[idx_a] += signal * shares_a
    positions[idx_b] += (-signal if beta > 0 else signal) * shares_b


def _pair_positions(prc_so_far: np.ndarray) -> np.ndarray:
    positions = np.zeros(N_INST, dtype=int)
    if not ENABLE_PAIRS:
        return positions

    prices_today = prc_so_far[:, -1]
    for pair_no, pair in enumerate(_active_pairs()):
        beta, _const, z, _spread_std = _pair_snapshot(prc_so_far, pair)
        if beta is None or z is None:
            _pair_signal[pair_no] = 0
            continue

        previous = _pair_signal[pair_no]
        if z > ENTRY_Z:
            signal = -1
        elif z < -ENTRY_Z:
            signal = 1
        elif abs(z) < EXIT_Z:
            signal = 0
        else:
            signal = previous

        _pair_signal[pair_no] = signal
        _add_pair_position(positions, prices_today, pair[0], pair[1], beta, signal)

    return positions


# ---------------------------------------------------------------------------
# Cross-sectional residual reversal
# ---------------------------------------------------------------------------
def _cross_sectional_target(prc_so_far: np.ndarray) -> np.ndarray:
    target = np.zeros(N_INST, dtype=int)
    n_days = prc_so_far.shape[1]
    min_days = CS_BETA_WINDOW + CS_RESIDUAL_DAYS + 2
    if not ENABLE_CROSS_SECTIONAL or n_days < min_days:
        return target

    returns = _daily_returns(prc_so_far)
    train = returns[:, -(CS_BETA_WINDOW + CS_RESIDUAL_DAYS):-CS_RESIDUAL_DAYS]
    signal_returns = returns[:, -CS_RESIDUAL_DAYS:]

    x = train[0]
    x_mean = float(x.mean())
    x_dev = x - x_mean
    x_var = float(np.dot(x_dev, x_dev))
    if x_var < 1e-12:
        return target

    excluded = _cross_excluded()
    algo_signal = signal_returns[0]
    residual_score = np.full(N_INST, np.nan)

    for idx in range(1, N_INST):
        if idx in excluded:
            continue
        y = train[idx]
        y_mean = float(y.mean())
        beta = float(np.dot(x_dev, y - y_mean) / x_var)
        alpha = y_mean - beta * x_mean
        residual_score[idx] = float(
            np.sum(signal_returns[idx] - (alpha + beta * algo_signal))
        )

    eligible = np.asarray(
        [
            idx for idx in range(1, N_INST)
            if idx not in excluded and np.isfinite(residual_score[idx])
        ],
        dtype=int,
    )
    if eligible.size < 2 * CS_K:
        return target

    ranked = eligible[np.argsort(residual_score[eligible])]
    longs = ranked[:CS_K]
    shorts = ranked[-CS_K:]
    prices_today = prc_so_far[:, -1]

    for idx in longs:
        shares = int(np.floor(CS_ALLOC_FRACTION * POSITION_LIMITS[idx] / prices_today[idx]))
        target[idx] = shares
    for idx in shorts:
        shares = int(np.floor(CS_ALLOC_FRACTION * POSITION_LIMITS[idx] / prices_today[idx]))
        target[idx] = -shares
    return target


def _cross_sectional_positions(prc_so_far: np.ndarray) -> np.ndarray:
    global _cs_target
    if not ENABLE_CROSS_SECTIONAL:
        _cs_target = np.zeros(N_INST, dtype=int)
        return _cs_target.copy()

    n_days = prc_so_far.shape[1]
    min_days = CS_BETA_WINDOW + CS_RESIDUAL_DAYS + 2
    if n_days < min_days:
        _cs_target = np.zeros(N_INST, dtype=int)
        return _cs_target.copy()

    if n_days % CS_REBALANCE_EVERY == 0:
        _cs_target = _cross_sectional_target(prc_so_far)
    return _cs_target.copy()


# ---------------------------------------------------------------------------
# Time-series momentum
# ---------------------------------------------------------------------------
def _timeseries_positions(prc_so_far: np.ndarray) -> np.ndarray:
    target = np.zeros(N_INST, dtype=int)
    if not ENABLE_TIMESERIES:
        return target

    prices_today = prc_so_far[:, -1]
    n_days = prc_so_far.shape[1]

    for idx, lookback, direction in _active_ts_signals():
        if n_days < lookback + 1:
            continue
        recent_return = float(prc_so_far[idx, -1] / prc_so_far[idx, -1 - lookback] - 1.0)
        signal = 1 if recent_return > 0 else (-1 if recent_return < 0 else 0)
        signal *= int(direction)  # +1 momentum, -1 reversal
        shares = int(np.floor(TS_ALLOC_FRACTION * POSITION_LIMITS[idx] / prices_today[idx]))
        target[idx] = signal * shares

    return target


# ---------------------------------------------------------------------------
# MHRM Ridge fallback
# ---------------------------------------------------------------------------
def _mhrm_ridge_prediction(prc_so_far: np.ndarray) -> float | None:
    """Predict MHRM's next daily return without look-ahead.

    Training observations use feature returns on day s to predict MHRM's
    return on day s+1.  The most recent feature-return vector then predicts
    the next unseen MHRM return.
    """
    n_days = prc_so_far.shape[1]
    if n_days < MHRM_MIN_OBS + 3:
        return None

    returns = _daily_returns(prc_so_far)
    if returns.shape[1] < MHRM_MIN_OBS + 1:
        return None

    feature_idx = np.asarray(MHRM_FEATURES, dtype=int)
    X = returns[feature_idx, :-1].T
    y = returns[MHRM_IDX, 1:]
    if X.shape[0] < MHRM_MIN_OBS:
        return None

    # Standardise using training information only.
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0, ddof=0)
    x_std = np.where(x_std < 1e-10, 1.0, x_std)
    Xs = (X - x_mean) / x_std

    y_mean = float(y.mean())
    yc = y - y_mean
    gram = Xs.T @ Xs + MHRM_RIDGE_ALPHA * np.eye(Xs.shape[1])
    rhs = Xs.T @ yc
    try:
        coef = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return None

    current_x = (returns[feature_idx, -1] - x_mean) / x_std
    prediction = float(y_mean + current_x @ coef)
    return prediction if np.isfinite(prediction) else None


def _mhrm_ridge_fallback_position(
    prc_so_far: np.ndarray,
    pair_positions: np.ndarray,
) -> np.ndarray:
    """Trade MHRM only while its MHRM/EAFC pair is completely flat."""
    target = np.zeros(N_INST, dtype=int)
    if not ENABLE_MHRM_RIDGE_FALLBACK:
        return target

    # Pair book has priority.  Do not alter either leg while it is active.
    if pair_positions[MHRM_IDX] != 0 or pair_positions[EAFC_IDX] != 0:
        return target

    prediction = _mhrm_ridge_prediction(prc_so_far)
    if prediction is None or abs(prediction) <= MHRM_PREDICTION_THRESHOLD:
        return target

    price = float(prc_so_far[MHRM_IDX, -1])
    if price <= 0:
        return target

    signal = 1 if prediction > 0 else -1
    shares = int(np.floor(
        MHRM_RIDGE_ALLOC_FRACTION * POSITION_LIMITS[MHRM_IDX] / price
    ))
    target[MHRM_IDX] = signal * shares
    return target


# ---------------------------------------------------------------------------
# Residual mean-reversion book (price level vs ALGO or vs the basket)
# ---------------------------------------------------------------------------
def _resid_positions(prc_so_far: np.ndarray) -> np.ndarray:
    target = np.zeros(N_INST, dtype=int)
    if not ENABLE_RESID_BOOKS:
        return target
    n_days = prc_so_far.shape[1]
    if n_days < RESID_Z_WINDOW + 2:
        return target

    allowed = _resid_assets()
    prices_today = prc_so_far[:, -1]
    algo = prc_so_far[0]
    basket = prc_so_far[1:].mean(axis=0)

    for idx in (ALGO_RESID_MEANREV + BASKET_RESID_MEANREV):
        if idx not in allowed:
            continue
        ref = algo if idx in ALGO_RESID_MEANREV else basket
        p = prc_so_far[idx]
        w_beta = min(RESID_BETA_WINDOW, n_days)
        beta, const = _ols_hedge(p[-w_beta:], ref[-w_beta:])
        if beta is None:
            continue
        w_z = min(RESID_Z_WINDOW, n_days)
        resid = p[-w_z:] - beta * ref[-w_z:] - const
        sigma = float(resid.std(ddof=0))
        if not np.isfinite(sigma) or sigma < 1e-9:
            continue
        z = float((resid[-1] - resid.mean()) / sigma)
        prev = _resid_signal.get(idx, 0)
        if z > RESID_ENTRY_Z:
            signal = -1
        elif z < -RESID_ENTRY_Z:
            signal = 1
        elif abs(z) < RESID_EXIT_Z:
            signal = 0
        else:
            signal = prev
        _resid_signal[idx] = signal
        if signal == 0:
            continue
        price = float(prices_today[idx])
        if price <= 0:
            continue
        target[idx] += signal * int(np.floor(
            RESID_ALLOC_FRACTION * POSITION_LIMITS[idx] / price))
    return target



# ---------------------------------------------------------------------------
# ALGO book: mean-revert ALGO against the equal-weighted basket of the other 50
# ---------------------------------------------------------------------------
def _algo_position(prc_so_far: np.ndarray) -> np.ndarray:
    global _algo_signal
    target = np.zeros(N_INST, dtype=int)
    if not ENABLE_ALGO_BOOK:
        return target
    n_days = prc_so_far.shape[1]
    if n_days < ALGO_Z_WINDOW + 2:
        return target

    algo = prc_so_far[0]
    basket = prc_so_far[1:].mean(axis=0)
    w_beta = min(ALGO_BETA_WINDOW, n_days)
    beta, const = _ols_hedge(algo[-w_beta:], basket[-w_beta:])
    if beta is None:
        return target

    w_z = min(ALGO_Z_WINDOW, n_days)
    resid = algo[-w_z:] - beta * basket[-w_z:] - const
    sigma = float(resid.std(ddof=0))
    if not np.isfinite(sigma) or sigma < 1e-9:
        return target
    z = float((resid[-1] - resid.mean()) / sigma)

    if z > ALGO_ENTRY_Z:
        signal = -1
    elif z < -ALGO_ENTRY_Z:
        signal = 1
    elif abs(z) < ALGO_EXIT_Z:
        signal = 0
    else:
        signal = _algo_signal
    _algo_signal = signal
    if signal == 0:
        return target

    price = float(prc_so_far[0, -1])
    if price <= 0:
        return target
    target[0] = signal * int(np.floor(ALGO_ALLOC_FRACTION * POSITION_LIMITS[0] / price))
    return target


# ---------------------------------------------------------------------------
# Live combination
# ---------------------------------------------------------------------------
def _run_strategy(prc_so_far: np.ndarray) -> np.ndarray:
    # Pair book is calculated first because MHRM/EAFC has priority over Ridge.
    pair_positions = _pair_positions(prc_so_far)

    positions = np.zeros(N_INST, dtype=int)
    positions += pair_positions
    positions += _cross_sectional_positions(prc_so_far)
    positions += _timeseries_positions(prc_so_far)
    positions += _resid_positions(prc_so_far)
    positions += _algo_position(prc_so_far)
    positions += _mhrm_ridge_fallback_position(prc_so_far, pair_positions)
    return positions


def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    """Required competition interface: return 51 integer target positions."""
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
