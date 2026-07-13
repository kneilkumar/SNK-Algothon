import numpy as np

# ---------------------------------------------------------------------------
# Stat-arb strategy: two independent cointegrated relationships, run
# side-by-side and summed into one position vector.
#
#   Leg A (pair):     HETT (idx 7)  vs  ULXY (idx 40)
#   Leg B (triplet):  OTCS (idx 6), SMAH (idx 10), ILVX (idx 46)
#
# Both legs share the same rolling window / entry / exit parameters, per
# instruction:
#   ROLLING_WINDOW = 93 days, ENTRY_Z = 1.5, EXIT_Z = 0.75
# (Leg A's window was originally derived from Leg A's own half-life; Leg B
# has a different half-life (train 3.7d / holdout 5.4d) but is intentionally
# kept on the same window/thresholds as instructed, rather than re-derived.)
#
# General spread definition for a leg with instruments idx = (i_1,...,i_k)
# and cointegrating vector v = (v_1,...,v_k):
#       S_t = sum_j  v_j * P_{i_j, t}
# (For Leg A, v = [1, -1.65382303] on [HETT, ULXY], matching the earlier
#  single-pair spec exactly. For Leg B, v = [1, 3.52312434, -5.83756638]
#  on [OTCS, SMAH, ILVX], as given.)
#
# Z-score: causal rolling mean/std of S_t over the trailing ROLLING_WINDOW
# days (no lookahead).
#
# Entry/exit rule (hysteresis), per leg, independently:
#   Flat  -> state = -1 ("short spread")  when z_t >  ENTRY_Z
#   Flat  -> state = +1 ("long spread")   when z_t < -ENTRY_Z
#   In position -> state = 0 (flatten)    when |z_t| < EXIT_Z
# State for "today" is obtained by replaying this rule over the ENTIRE
# z-score history from scratch on every call (stateless / idempotent -- see
# note below).
#
# Position sizing, per leg, independently: solve for the largest scale N
# such that every leg-instrument stays within +/- $10k:
#       N = min over j of  DOLLAR_CAP / (|v_j| * price_j)
# then position_{i_j} = round(state * v_j * N).
# This means whichever instrument is tightest binds exactly at (or just
# under) $10k, and every other instrument in that leg is scaled by the SAME
# N, so the cointegrating ratio is preserved exactly (no independent-cap
# distortion). This mirrors the design decision made for the HETT/ULXY pair.
#
# The two legs touch disjoint instruments (7, 40 vs 6, 10, 46), so their
# positions are simply summed with no interaction/conflict.
#
# State is recomputed from scratch each call (rather than relying on a
# saved "were we in a trade yesterday" flag) so the function is correct
# regardless of how/how often the grader invokes getMyPosition.
# ---------------------------------------------------------------------------

ROLLING_WINDOW = 93
ENTRY_Z = 1.5
EXIT_Z = 0.75
DOLLAR_CAP = 10000.0

LEGS = [
    {
        "name": "HETT/ULXY",
        "indices": (7, 40),
        "coeffs": (1.0, -1.65382303),
    },
    {
        "name": "OTCS/SMAH/ILVX",
        "indices": (6, 10, 46),
        "coeffs": (1.0, 3.52312434, -5.83756638),
    },
]


def _rolling_mean_std(x, window):
    """Causal rolling mean/std (no lookahead): value at index i uses
    x[i-window+1 : i+1]. First (window-1) entries are NaN (insufficient
    history)."""
    n = len(x)
    mean = np.full(n, np.nan)
    std = np.full(n, np.nan)
    if n < window:
        return mean, std
    csum = np.cumsum(x)
    csum2 = np.cumsum(x * x)
    for i in range(window - 1, n):
        if i == window - 1:
            s = csum[i]
            s2 = csum2[i]
        else:
            s = csum[i] - csum[i - window]
            s2 = csum2[i] - csum2[i - window]
        m = s / window
        var = max(s2 / window - m * m, 0.0)
        mean[i] = m
        std[i] = np.sqrt(var)
    return mean, std


def _simulate_state(z):
    """Replay entry/exit hysteresis rule over the full z-score history and
    return the final state: +1 = long spread, -1 = short spread, 0 = flat."""
    state = 0
    for zt in z:
        if np.isnan(zt):
            continue
        if state == 0:
            if zt > ENTRY_Z:
                state = -1
            elif zt < -ENTRY_Z:
                state = 1
        else:
            if abs(zt) < EXIT_Z:
                state = 0
    return state


def _leg_positions(prcSoFar, indices, coeffs):
    """Compute the integer position contribution of a single leg
    (pair or triplet) given its instrument indices and cointegrating
    vector."""
    n_instruments, n_days = prcSoFar.shape
    contrib = np.zeros(n_instruments, dtype=int)

    if n_days < ROLLING_WINDOW:
        return contrib

    prices = [prcSoFar[idx, :] for idx in indices]
    coeffs = np.asarray(coeffs, dtype=float)

    spread = np.zeros(n_days)
    for p, v in zip(prices, coeffs):
        spread += v * p

    mean, std = _rolling_mean_std(spread, ROLLING_WINDOW)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (spread - mean) / std
    z[std == 0] = 0.0

    state = _simulate_state(z)
    if state == 0:
        return contrib

    last_prices = np.array([p[-1] for p in prices])
    if np.any(last_prices <= 0):
        return contrib

    # Largest common scale N keeping every leg-instrument within the
    # dollar cap.
    n_candidates = DOLLAR_CAP / (np.abs(coeffs) * last_prices)
    n_scale = np.min(n_candidates)

    shares = np.floor(np.abs(coeffs) * n_scale).astype(int)
    if np.any(shares <= 0):
        return contrib

    signs = np.sign(state * coeffs)
    for idx, sh, sg in zip(indices, shares, signs):
        contrib[idx] = int(sg) * sh

    return contrib


def getMyPosition(prcSoFar):
    prcSoFar = np.asarray(prcSoFar, dtype=float)
    n_instruments, n_days = prcSoFar.shape

    positions = np.zeros(n_instruments, dtype=int)

    for leg in LEGS:
        positions += _leg_positions(prcSoFar, leg["indices"], leg["coeffs"])

    return positions
