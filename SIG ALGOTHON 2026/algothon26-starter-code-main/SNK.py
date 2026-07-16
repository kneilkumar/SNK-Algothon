import numpy as np

# ---------------------------------------------------------------------------
# Two sleeves, one book:
#   1) HETT/ULXY cointegrated pair trade (unchanged from the previous version)
#   2) Avellaneda-Lee PCA/OU statistical arbitrage across the rest of the
#      universe (new -- adapted from a one-shot offline research script into
#      a fully causal, stateless function suitable for getMyPosition)
#
# Sleeve 2 is additive: it never touches HETT/ULXY (those are reserved for
# the pair trade so the two sleeves can never fight each other for the same
# instrument), but it DOES use HETT/ULXY's returns as part of the PCA
# cross-section, since excluding them there would throw away information
# about market-wide factors for no benefit.
#
# ===========================================================================
# WHY THE AVELLANEDA-LEE IMPLEMENTATION DIFFERS FROM THE ORIGINAL RESEARCH
# SCRIPT YOU PASTED -- READ THIS BEFORE ASSUMING A DEVIATION IS A BUG
# ===========================================================================
# The pasted script was a ONE-SHOT, OFFLINE analysis: it loaded a static
# prices.csv, sliced the last 252 days once, ran sklearn's PCA and
# statsmodels' adfuller once, and printed a single day's signal. That's fine
# for research, but getMyPosition is called fresh every day with the full
# price history so far and must return a POSITION, not a printout -- and it
# has no persistent state between calls. Three consequences:
#
# 1. NO SKLEARN / STATSMODELS. This file only ever imports numpy (matching
#    the rest of SNK.py, and matching how the pair sleeve already does its
#    own OLS AR(1) fit by hand). PCA is done via eigendecomposition of the
#    correlation matrix (np.linalg.eigh) instead of sklearn.decomposition.PCA
#    -- mathematically identical for this purpose. The ADF stationarity test
#    is replaced with the SAME criterion the pair sleeve already uses: the
#    AR(1) coefficient must lie in (0,1) (i.e. mean-reverting), plus a
#    half-life sanity band. This is a materially different (weaker) test
#    than ADF -- it doesn't have ADF's asymptotic distribution or unit-root
#    null. If you have statsmodels available in your actual runtime and want
#    the real ADF test back, say so and I'll wire it in with a fallback.
#
# 2. FULLY CAUSAL, RE-FIT EVERY DAY. PCA + the per-instrument regression
#    + the OU fit are re-estimated from scratch for every day t using only
#    a trailing window of returns ending at t (never looking ahead), exactly
#    like the pair sleeve's rolling half-life. This is the expensive part:
#    it's an O(n_days) loop of PCA-eigh + a single vectorized multi-output
#    lstsq call (all instruments' regressions solved simultaneously) + a
#    fully vectorized cross-sectional AR(1) fit. No per-instrument Python
#    loop inside the day loop -- the whole cross-section is fit in one shot
#    each day.
#
# 3. HYSTERESIS, NOT A ONE-OFF SIGNAL. The pasted script just classified
#    "today's s-score" as LONG/SHORT/EXIT/HOLD, implicitly relying on some
#    external process to remember whether you were already in the trade.
#    getMyPosition has no such external memory, so -- exactly like the pair
#    sleeve's _simulate_state -- each instrument's entry/exit history is
#    replayed day-by-day from its full s-score history to recover today's
#    true state (flat / long / short). This is more expensive than a single
#    lookup but it's the only way to get a correct position out of a
#    stateless function.
#
# ===========================================================================
# POSITION SIZING RULES (as specified)
# ===========================================================================
# - MAX_DOLLAR_PER_INSTRUMENT = $10,000: no single instrument's position
#   (in either sleeve) may exceed this in absolute dollar terms.
# - MIN_TOTAL_DOLLAR_EXPOSURE = $25,000: if the pair sleeve + AL sleeve's
#   genuine signals don't add up to at least this much gross exposure, the
#   book is topped up with the AL universe's strongest not-yet-triggered
#   s-scores (largest |s|, still capped at $10k each) until the floor is
#   met or candidates run out. This is a deliberate, clearly-separate
#   "filler" step -- see _topup_to_minimum_exposure -- because there's no
#   principled way to manufacture $25k of exposure out of instruments with
#   NO signal at all without inventing a rule; ranking by |s-score| among
#   screened (mean-reverting) instruments is the least-arbitrary fallback
#   available. If you'd rather this floor be a hard requirement even when
#   there are zero screened candidates (impossible to satisfy honestly), or
#   you want a different fallback ranking, tell me and I'll change it.
# ===========================================================================

HETT_IDX = 7
ULXY_IDX = 40
BETA = 1.65382303

# --- shared position sizing rules ---
MAX_DOLLAR_PER_INSTRUMENT = 10000.0   # per-instrument cap, both sleeves
MIN_TOTAL_DOLLAR_EXPOSURE = 25000.0   # whole-book floor (see note above)

# ---------------------------------------------------------------------------
# SLEEVE 1: HETT/ULXY pair trade -- UNCHANGED logic from the previous version
# ---------------------------------------------------------------------------

# --- half-life estimation (rolling, causal) ---
HL_LOOKBACK = 90          # trailing days used to fit the AR(1) half-life estimate
HL_MIN_FIT_LEN = 30       # minimum days of spread history before we attempt a fit
HALF_LIFE_DEFAULT = 5.8   # prior used only before any valid fit exists yet (original train estimate)
HALF_LIFE_MIN = 1.0       # clip: half-life can't be estimated below this (numerically meaningless)
HALF_LIFE_MAX = 60.0      # clip: half-life can't be estimated above this (effectively "not mean-reverting")

# --- window derived from half-life ---
N_INDEPENDENT_SAMPLES = 8   # same convention as the original window_from_half_life()
WINDOW_MIN = 20             # floor: need at least this many points for a sane mean/std
WINDOW_MAX = 200            # ceiling: cap on how sluggish the window is allowed to get

# --- entry/exit thresholds: fixed, NOT derived from half-life ---
ENTRY_Z = 1.5
EXIT_Z = 0.75


def _fit_half_life(spread_slice):
    """OLS AR(1) fit S_t = a + b*S_{t-1} + eps on a 1D slice.
    Returns half-life in days, or NaN if the fit is degenerate
    (b outside (0,1), i.e. not mean-reverting on this slice)."""
    x = spread_slice[:-1]
    y = spread_slice[1:]
    if len(x) < 2 or np.std(x) == 0:
        return float("nan")
    b, a = np.polyfit(x, y, 1)
    if not (0.0 < b < 1.0):
        return float("nan")
    return -np.log(2.0) / np.log(b)


def _rolling_half_life_series(spread):
    """Causal rolling half-life estimate for every day t, using only
    spread[max(0, t-HL_LOOKBACK+1) : t+1]. Before HL_MIN_FIT_LEN days of
    history exist, or whenever a fit is degenerate, carries forward the
    last valid estimate (falling back to HALF_LIFE_DEFAULT if none yet
    exists). Every value is clipped to [HALF_LIFE_MIN, HALF_LIFE_MAX]."""
    n = len(spread)
    half_life = np.full(n, np.nan)
    last_valid = HALF_LIFE_DEFAULT
    for t in range(n):
        start = max(0, t - HL_LOOKBACK + 1)
        seg = spread[start:t + 1]
        if len(seg) >= HL_MIN_FIT_LEN:
            hl = _fit_half_life(seg)
            if not np.isnan(hl):
                hl_clipped = min(max(hl, HALF_LIFE_MIN), HALF_LIFE_MAX)
                last_valid = hl_clipped
        half_life[t] = last_valid
    return half_life


def _dynamic_window_series(half_life, n_days):
    """window_t = round(N_INDEPENDENT_SAMPLES * 2 * half_life_t), clipped
    to [WINDOW_MIN, WINDOW_MAX], and further clipped so it never exceeds
    the amount of history available on day t (t+1 days)."""
    raw = np.round(N_INDEPENDENT_SAMPLES * 2.0 * half_life).astype(int)
    raw = np.clip(raw, WINDOW_MIN, WINDOW_MAX)
    available = np.arange(1, n_days + 1)  # day t (0-indexed) has t+1 days of history
    window = np.minimum(raw, available)
    return window


def _dynamic_zscore_series(spread, window):
    """z_t computed with a per-day rolling window length (window[t]),
    using only spread[t-window[t]+1 : t+1] (causal). NaN wherever there
    isn't even WINDOW_MIN days of history yet."""
    n = len(spread)
    z = np.full(n, np.nan)
    for t in range(n):
        w = window[t]
        if w < WINDOW_MIN:
            continue
        seg = spread[t - w + 1:t + 1]
        mu = seg.mean()
        sigma = seg.std()
        z[t] = (spread[t] - mu) / sigma if sigma > 0 else 0.0
    return z


def _simulate_state_generic(scores, entry, exit_):
    """Generic entry/exit hysteresis replay: +1 = long, -1 = short,
    0 = flat, as of the last point in `scores`. NaNs are skipped (treated
    as "no update today", not as "exit")."""
    state = 0
    for s in scores:
        if np.isnan(s):
            continue
        if state == 0:
            if s > entry:
                state = -1
            elif s < -entry:
                state = 1
        else:
            if abs(s) < exit_:
                state = 0
    return state


def _simulate_state(z):
    """Pair-sleeve hysteresis replay, kept as its own named function for
    backward compatibility. +1 = long spread (long HETT, short ULXY),
    -1 = short spread, 0 = flat."""
    return _simulate_state_generic(z, ENTRY_Z, EXIT_Z)


def _pair_sleeve_positions(prcSoFar, beta, n_instruments, n_days):
    positions = np.zeros(n_instruments, dtype=int)
    if n_days < WINDOW_MIN:
        return positions

    p_hett = prcSoFar[HETT_IDX, :]
    p_ulxy = prcSoFar[ULXY_IDX, :]
    spread = p_hett - beta * p_ulxy

    half_life = _rolling_half_life_series(spread)
    window = _dynamic_window_series(half_life, n_days)
    z = _dynamic_zscore_series(spread, window)

    state = _simulate_state(z)
    if state == 0:
        return positions

    price_hett = p_hett[-1]
    price_ulxy = p_ulxy[-1]
    if price_hett <= 0 or price_ulxy <= 0:
        return positions

    n_from_hett = MAX_DOLLAR_PER_INSTRUMENT / price_hett
    n_from_ulxy = MAX_DOLLAR_PER_INSTRUMENT / (beta * price_ulxy)
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


# ---------------------------------------------------------------------------
# SLEEVE 2: Avellaneda-Lee PCA/OU stat-arb across the rest of the universe
# ---------------------------------------------------------------------------

AL_RETURNS_LOOKBACK = 252   # trailing return days used to fit PCA + OU each day
AL_MIN_LOOKBACK = 60        # minimum trailing return days before we attempt a fit at all
AL_N_COMPONENTS = 12        # target number of eigenportfolios (clipped to what's feasible)

AL_MIN_HALF_LIFE = 2.0      # screening band, same convention as the pasted script
AL_MAX_HALF_LIFE = 30.0

AL_ENTRY_S = 1.25           # fixed, taken directly from the pasted script
AL_EXIT_S = 0.75


def _causal_returns(prcSoFar):
    """Simple returns for every instrument, causal by construction:
    rets[:, t] only uses prcSoFar[:, t] and prcSoFar[:, t+1]."""
    p = prcSoFar
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = (p[:, 1:] - p[:, :-1]) / p[:, :-1]
    rets = np.where(np.isfinite(rets), rets, 0.0)
    return rets


def _standardize_columns(window_returns):
    """window_returns: (n_instr, w) -> returns (w, n_instr) z-scored per
    instrument over the window (this is `returns_normalised` in the pasted
    script, just transposed to the (obs, features) convention PCA wants)."""
    mean_r = window_returns.mean(axis=1, keepdims=True)
    std_r = window_returns.std(axis=1, keepdims=True)
    std_safe = np.where(std_r == 0, 1.0, std_r)
    norm = (window_returns - mean_r) / std_safe
    return norm.T  # (w, n_instr)


def _pca_eigenportfolio_returns(norm, n_components):
    """norm: (w, n_instr) standardized returns. Returns (w, k) eigenportfolio
    return series -- equivalent to sklearn's PCA(n_components=k).fit_transform
    on the same input, via eigendecomposition of the correlation matrix
    instead of sklearn's SVD-based implementation."""
    w, n_instr = norm.shape
    k = max(1, min(n_components, n_instr - 1, w - 1))
    cov = np.cov(norm, rowvar=False)
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)   # ascending order
    order = np.argsort(eigvals)[::-1]
    top_vecs = eigvecs[:, order[:k]]         # (n_instr, k)
    eigen_returns = norm @ top_vecs          # (w, k)
    return eigen_returns


def _residual_price_paths(window_returns, eigen_returns):
    """window_returns: (n_instr, w) RAW (non-normalized) returns.
    eigen_returns: (w, k). For every instrument, regress its raw returns on
    the eigenportfolio returns (with intercept), then cumsum the residual
    returns into a synthetic residual "price" path -- all instruments solved
    in a single vectorized lstsq call (this is the multi-output form of the
    per-instrument loop in the pasted script). Returns (w, n_instr)."""
    w = window_returns.shape[1]
    X = np.column_stack([np.ones(w), eigen_returns])       # (w, k+1)
    Y = window_returns.T                                    # (w, n_instr)
    coefs, *_ = np.linalg.lstsq(X, Y, rcond=None)           # (k+1, n_instr)
    fitted = X @ coefs                                      # (w, n_instr)
    resid_returns = Y - fitted
    resid_price = np.cumsum(resid_returns, axis=0)          # (w, n_instr)
    return resid_price


def _fit_ou_params_cross_section(resid_price):
    """resid_price: (w, n_instr). Fully vectorized cross-sectional AR(1)/OU
    fit -- the multi-instrument form of fit_OU_proc() in the pasted script.
    Returns (theta, mu, sigma, half_life), each shape (n_instr,), NaN where
    the fit is degenerate (AR(1) coefficient outside (0,1) or non-positive
    variance)."""
    x = resid_price[:-1, :]
    y = resid_price[1:, :]
    n_obs = x.shape[0]

    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    x_dm = x - x_mean
    y_dm = y - y_mean
    var_x = (x_dm ** 2).mean(axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        phi = np.where(var_x > 0, (x_dm * y_dm).mean(axis=0) / var_x, np.nan)
    intercept = y_mean - phi * x_mean
    fitted = phi[None, :] * x + intercept[None, :]
    errors = y - fitted

    ddof = 2
    if n_obs <= ddof:
        var_error = np.full(x.shape[1], np.nan)
    else:
        var_error = (errors ** 2).sum(axis=0) / (n_obs - ddof)

    valid = np.isfinite(phi) & (phi > 0.0) & (phi < 1.0)
    phi_safe = np.where(valid, phi, np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        theta = -np.log(phi_safe) / 1.0
        mu = intercept / (1.0 - phi_safe)
        sigma_sq = 2.0 * theta * var_error / (1.0 - phi_safe ** 2)
        sigma = np.sqrt(np.where(sigma_sq > 0, sigma_sq, np.nan))
        half_life = np.log(2.0) / theta

    return theta, mu, sigma, half_life


def _al_s_score_history(prcSoFar, n_instruments, n_days):
    """Causal, day-by-day s-score history for every instrument (all
    instruments are included in the PCA cross-section; trading decisions
    later exclude HETT/ULXY). s_hist[i, t] uses only returns up to day t+1
    (i.e. prcSoFar[:, :t+2]) -- never looks ahead."""
    rets = _causal_returns(prcSoFar)             # (n_instr, n_days-1)
    n_ret_days = rets.shape[1]
    s_hist = np.full((n_instruments, n_ret_days), np.nan)

    for t in range(AL_MIN_LOOKBACK - 1, n_ret_days):
        start = max(0, t - AL_RETURNS_LOOKBACK + 1)
        window_returns = rets[:, start:t + 1]     # (n_instr, w)
        if window_returns.shape[1] < AL_MIN_LOOKBACK:
            continue

        norm = _standardize_columns(window_returns)
        eigen_returns = _pca_eigenportfolio_returns(norm, AL_N_COMPONENTS)
        resid_price = _residual_price_paths(window_returns, eigen_returns)  # (w, n_instr)

        theta, mu, sigma, half_life = _fit_ou_params_cross_section(resid_price)
        sigma_eq = sigma / np.sqrt(2.0 * theta)

        band_ok = (half_life >= AL_MIN_HALF_LIFE) & (half_life <= AL_MAX_HALF_LIFE)
        with np.errstate(invalid="ignore"):
            s_now = (resid_price[-1, :] - mu) / sigma_eq
        s_now = np.where(band_ok & np.isfinite(s_now), s_now, np.nan)

        s_hist[:, t] = s_now

    return s_hist


def _al_sleeve_positions(prcSoFar, n_instruments, n_days):
    """Returns (positions, candidates): positions is an (n_instruments,)
    array with the AL sleeve's target positions for every instrument except
    HETT/ULXY (always zero for those two -- reserved for the pair sleeve).
    candidates is a list of (instrument_idx, latest_s_score) for instruments
    that are currently flat but have a valid, screened s-score today --
    used by the minimum-exposure top-up, not by the AL sleeve itself."""
    positions = np.zeros(n_instruments, dtype=int)
    candidates = []

    n_ret_days = n_days - 1
    if n_ret_days < AL_MIN_LOOKBACK:
        return positions, candidates

    s_hist = _al_s_score_history(prcSoFar, n_instruments, n_days)
    last_prices = prcSoFar[:, -1]

    for inst in range(n_instruments):
        if inst in (HETT_IDX, ULXY_IDX):
            continue

        s_series = s_hist[inst, :]
        state = _simulate_state_generic(s_series, AL_ENTRY_S, AL_EXIT_S)
        s_last = s_series[-1]

        if state != 0:
            price = last_prices[inst]
            if price > 0:
                n_shares = int(np.floor(MAX_DOLLAR_PER_INSTRUMENT / price))
                if n_shares > 0:
                    positions[inst] = n_shares if state == 1 else -n_shares
        elif not np.isnan(s_last):
            candidates.append((inst, s_last))

    return positions, candidates


def _topup_to_minimum_exposure(positions, last_prices, candidates):
    """If total gross dollar exposure across the whole book is below
    MIN_TOTAL_DOLLAR_EXPOSURE, add positions in the AL universe's strongest
    not-yet-triggered s-scores (largest |s|, still respecting the $10k
    per-instrument cap and direction implied by the sign of s) until the
    floor is met or candidates run out. See the module docstring for why
    this specific fallback was chosen."""
    gross = np.sum(np.abs(positions) * last_prices)
    if gross >= MIN_TOTAL_DOLLAR_EXPOSURE or not candidates:
        return positions

    ranked = sorted(candidates, key=lambda c: abs(c[1]), reverse=True)
    for inst, s_last in ranked:
        if gross >= MIN_TOTAL_DOLLAR_EXPOSURE:
            break
        if positions[inst] != 0:
            continue
        price = last_prices[inst]
        if price <= 0:
            continue
        n_shares = int(np.floor(MAX_DOLLAR_PER_INSTRUMENT / price))
        if n_shares <= 0:
            continue
        direction = 1 if s_last < 0 else -1   # below OU mean -> long, above -> short
        positions[inst] = direction * n_shares
        gross += n_shares * price

    return positions


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

def getMyPosition(prcSoFar, beta=BETA):
    """beta exposed as a keyword purely for reuse in sensitivity-testing
    scripts -- the competition harness will always call this with just
    prcSoFar, and it will use the fitted BETA by default."""
    prcSoFar = np.asarray(prcSoFar, dtype=float)
    n_instruments, n_days = prcSoFar.shape

    positions = _pair_sleeve_positions(prcSoFar, beta, n_instruments, n_days)

    al_positions, candidates = _al_sleeve_positions(prcSoFar, n_instruments, n_days)
    positions = positions + al_positions

    positions = _topup_to_minimum_exposure(positions, prcSoFar[:, -1], candidates)

    return positions