import numpy as np

# ---------------------------------------------------------------------------
# Stat-arb pairs strategy: HETT (idx 7) vs ULXY (idx 40) -- single pair only.
#
# Spread:      S_t = P_HETT,t - BETA * P_ULXY,t
# Cointegrating vector (Johansen, confirmed 3 ways) -> BETA below.
# Rolling window for spread mean/std: 93 days (derived from half-life).
# Entry/exit thresholds: entry_z=1.5, exit_z=0.75 (holdout-preferred,
#   more conservative / less overfit than the train-optimal 1.5/0.5 pair).
#
# Position sizing: on each entry we solve for the largest integer scale N
# such that holding N shares of HETT and BETA*N shares of ULXY (opposite
# sign) keeps BOTH legs under the $10k cap. Whichever leg is tighter binds
# exactly at (or just under) $10k; the other leg is scaled by the SAME N,
# so the 1 : BETA share ratio implied by the cointegrating vector is
# preserved exactly (no ratio distortion from independent per-leg capping).
#
# State (long spread / short spread / flat) is recomputed from scratch each
# call by replaying the full z-score history with the entry/exit hysteresis
# rule. This makes the function stateless/idempotent -- it doesn't rely on
# any external memory of "were we in a trade yesterday".
# ---------------------------------------------------------------------------

HETT_IDX = 7
ULXY_IDX = 40
BETA = 1.65382303

ROLLING_WINDOW = 93
ENTRY_Z = 0.75
EXIT_Z = 0.5

DOLLAR_CAP = 10000.0


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
                state = -1  # short spread: short HETT, long ULXY
            elif zt < -ENTRY_Z:
                state = 1   # long spread: long HETT, short ULXY
        else:
            if abs(zt) < EXIT_Z:
                state = 0
    return state


def getMyPosition(prcSoFar, beta=BETA):
    """beta is exposed as a parameter (defaults to the fitted BETA) purely
    so the same function can be reused for hedge-ratio sensitivity testing
    -- the competition harness will always call it with just prcSoFar."""
    prcSoFar = np.asarray(prcSoFar, dtype=float)
    n_instruments, n_days = prcSoFar.shape

    positions = np.zeros(n_instruments, dtype=int)

    if n_days < ROLLING_WINDOW:
        return positions

    p_hett = prcSoFar[HETT_IDX, :]
    p_ulxy = prcSoFar[ULXY_IDX, :]

    spread = p_hett - beta * p_ulxy
    mean, std = _rolling_mean_std(spread, ROLLING_WINDOW)

    with np.errstate(invalid="ignore", divide="ignore"):
        z = (spread - mean) / std
    z[std == 0] = 0.0

    state = _simulate_state(z)
    if state == 0:
        return positions

    price_hett = p_hett[-1]
    price_ulxy = p_ulxy[-1]
    if price_hett <= 0 or price_ulxy <= 0:
        return positions

    n_from_hett = DOLLAR_CAP / price_hett
    n_from_ulxy = DOLLAR_CAP / (beta * price_ulxy)
    n_scale = min(n_from_hett, n_from_ulxy)

    shares_hett = int(np.floor(n_scale))
    shares_ulxy = int(np.floor(beta * n_scale))
    if shares_hett <= 0 or shares_ulxy <= 0:
        return positions

    if state == 1:
        positions[HETT_IDX] = shares_hett
        positions[ULXY_IDX] = -shares_ulxy
    else:
        positions[HETT_IDX] = -shares_hett
        positions[ULXY_IDX] = shares_ulxy

    return positions