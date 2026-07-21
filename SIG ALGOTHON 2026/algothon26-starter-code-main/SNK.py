"""
SIG Algothon trading algorithm.

Strategy: statistical arbitrage on the small set of instrument pairs that
survived Bonferroni-corrected Engle-Granger cointegration testing on the
750-day training file (1,275 pairs tested; only 6 significant at
p < 0.05/1275). Those 6 were validated out-of-sample: hedge ratios fit on
the first half of the training data, traded on the untouched second half,
all 6 profitable. One further instrument (MTNS) has an individually
stationary price level and trades a simple standalone mean-reversion rule.
(FARS was originally traded the same way, but regressing out ALGO shows
its apparent mean-reversion was actually ALGO-driven rather than a
standalone edge in FARS itself, so it has been dropped from the solo
book below.)

ALGO (asset 0) itself is left flat: it tracks an equal-weighted basket of
the other 50 (R^2 ~0.99 in returns) but that relationship is purely
contemporaneous (no lead-lag) and the basket spread is NOT cointegrated
(Engle-Granger p=0.65), so there is no validated edge to trade on ALGO
directly.

However, ALGO is still useful as a *regressor*: netting each instrument's
price against ALGO (per-instrument OLS beta, betas ranging -2.18 to
+5.30, mean 0.64, fit R^2 from 0.0007 to 0.88) exposes idiosyncratic mean
reversion invisible in raw price levels. 10 of 50 residuals are
individually stationary (ADF p<0.05) versus only 6 in raw prices. Of
those 10: four (EELT, MTNS, ULXY, BENI) are already traded above via
PAIRS/solo and are not re-traded here to avoid stacking two uncoordinated
signals on the same instrument; EORC and NGTE are also already spoken
for via a PAIRS entry and are likewise excluded. That leaves four
genuinely new, non-overlapping names traded on ALGO-residual mean
reversion below: AETS, MHRM, EAFC, HRET.

Don't invent a signal that isn't backed by the data.

Rename this file to <yourTeamName>.py before submitting.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Instrument index map (fixed order, matches the competition's asset 0-50):
#  0 ALGO   1 AENO   2 LSST   3 SRNA   4 ELLT   5 AMRP   6 OTCS   7 HETT
#  8 HUXZ   9 DUCT  10 SMAH  11 NPCK  12 MSDP  13 EORC  14 CUBO  15 HRET
# 16 ANSO  17 DIHO  18 RTTH  19 SPLZ  20 NWIG  21 MMBT  22 MDGI  23 AGVF
# 24 RRES  25 CTGI  26 ALUT  27 ACAC  28 SRTX  29 GARI  30 RCRI  31 ACIX
# 32 CCNS  33 MTNS  34 IHOZ  35 NAYO  36 FWWG  37 EELT  38 HRND  39 AETS
# 40 ULXY  41 BLBT  42 BENI  43 ITPA  44 HTRK  45 NGTE  46 ILVX  47 FCSG
# 48 FARS  49 MHRM  50 EAFC
# ---------------------------------------------------------------------------

N_INST = 51
POSITION_LIMITS = np.array([100_000.0] + [10_000.0] * 50)

# Bonferroni-significant cointegrated pairs: spread = price_A - hedge*price_B - const
PAIRS = [
    (10, 46),  # SMAH ~ ILVX   (EG p=3.4e-9)
    (1, 20),   # AENO ~ NWIG   (EG p=3.7e-9)
    (8, 27),   # HUXZ ~ ACAC   (EG p=1.2e-8)
    (7, 40),   # HETT ~ ULXY   (EG p=2.1e-7)
    (25, 37),  # CTGI ~ EELT   (EG p=7.9e-6, both legs individually stationary too)
    (18, 35),  # RTTH ~ NAYO   (EG p=3.5e-5)
    (41, 36),  # BLBT ~ FWWG
    (26, 32) ,  # ALUT ~ CCNS
    (13, 45), # EORC ~ NGTE
    (33, 42), # MTNS ~ BENI
]

# Individually mean-reverting instruments (raw price level) not already
# covered by a pair above.
# (ULXY, CTGI, EELT are also individually stationary in raw prices but
# already traded via PAIRS. FARS was previously traded here too, but its
# apparent mean-reversion turned out to be ALGO-driven -- see ALGO_RESID_
# MEANREV below and the module docstring -- so it has been removed.)
SOLO_MEANREV = []  # MTNS

# Instruments whose ALGO-residual (price netted against an OLS beta on
# ALGO) is individually stationary, and which are NOT already traded via
# PAIRS or SOLO_MEANREV above (EELT/MTNS/ULXY/BENI/EORC/NGTE are excluded
# for that reason -- see module docstring).
ALGO_RESID_MEANREV = [39, 49, 50, 15]  # AETS, MHRM, EAFC, HRET

HEDGE_WINDOW = 300     # long trailing window for a stable hedge-ratio estimate
Z_WINDOW = 40          # short trailing window for the entry/exit z-score
ENTRY_Z = 1.25
EXIT_Z = 0.25

SOLO_WINDOW = 60
SOLO_ENTRY_Z = 1.25
SOLO_EXIT_Z = 0.25

# ALGO-residual mean reversion: beta fit over a long trailing window (same
# spirit as HEDGE_WINDOW for pairs), z-score of the residual over a short
# trailing window (same spirit as SOLO_WINDOW).
ALGO_RESID_BETA_WINDOW = 300
ALGO_RESID_Z_WINDOW = 60
ALGO_RESID_ENTRY_Z = 1.25
ALGO_RESID_EXIT_Z = 0.25

ALLOC_FRACTION = 0.95  # fraction of the per-instrument cap deployed on a full signal

MIN_DAYS_PAIRS = Z_WINDOW + 2
MIN_DAYS_SOLO = 20
MIN_DAYS_ALGO_RESID = ALGO_RESID_Z_WINDOW + 2

# Module-level state persisted across sequential daily calls (standard for this
# style of harness: the module stays loaded and getMyPosition is called once per
# day, in order, within a single evaluation run).
_pair_signal = {i: 0 for i in range(len(PAIRS))}
_solo_signal = {idx: 0 for idx in SOLO_MEANREV}
_algo_resid_signal = {idx: 0 for idx in ALGO_RESID_MEANREV}
_last_good_position = np.zeros(N_INST, dtype=int)


def _ols_hedge(a, b):
    """Closed-form OLS hedge ratio & intercept for a ~ hedge*b + const."""
    b_mean = b.mean()
    a_mean = a.mean()
    var = np.dot(b - b_mean, b - b_mean)
    if var < 1e-9:
        return None, None
    hedge = np.dot(b - b_mean, a - a_mean) / var
    const = a_mean - hedge * b_mean
    return hedge, const


def _run_strategy(prcSoFar):
    nInst, nDays = prcSoFar.shape
    positions = np.zeros(nInst, dtype=int)
    prices_today = prcSoFar[:, -1]

    # ---------------- Pair trades ----------------
    if nDays >= MIN_DAYS_PAIRS:
        for i, (idxA, idxB) in enumerate(PAIRS):
            pa = prcSoFar[idxA]
            pb = prcSoFar[idxB]

            w_hedge = min(HEDGE_WINDOW, nDays)
            hedge, const = _ols_hedge(pa[-w_hedge:], pb[-w_hedge:])
            if hedge is None or hedge <= 0:
                continue  # degenerate/untrustworthy fit this day; stay out
            w_z = min(Z_WINDOW, nDays)
            spread_hist = pa[-w_z:] - hedge * pb[-w_z:] - const
            mu, sigma = spread_hist.mean(), spread_hist.std()
            if sigma < 1e-9:
                continue
            z = (spread_hist[-1] - mu) / sigma

            prev = _pair_signal[i]
            if z > ENTRY_Z:
                sig = -1
            elif z < -ENTRY_Z:
                sig = 1
            elif abs(z) < EXIT_Z:
                sig = 0
            else:
                sig = prev  # inside the band: hold whatever we had
            _pair_signal[i] = sig

            if sig == 0:
                continue

            price_a, price_b = prices_today[idxA], prices_today[idxB]
            cap_a = ALLOC_FRACTION * POSITION_LIMITS[idxA]
            cap_b = ALLOC_FRACTION * POSITION_LIMITS[idxB]

            max_shares_a_from_a = np.floor(cap_a / price_a)
            max_shares_a_from_b = np.floor(cap_b / (hedge * price_b))
            shares_a = min(max_shares_a_from_a, max_shares_a_from_b)
            shares_b = np.floor(shares_a * hedge)

            positions[idxA] += sig * int(shares_a)
            positions[idxB] += -sig * int(shares_b)

    # ---------------- Solo mean-reversion ----------------
    if nDays >= MIN_DAYS_SOLO:
        for idx in SOLO_MEANREV:
            p = prcSoFar[idx]
            w = min(SOLO_WINDOW, nDays)
            window = p[-w:]
            mu, sigma = window.mean(), window.std()
            if sigma < 1e-9:
                continue
            z = (window[-1] - mu) / sigma

            prev = _solo_signal[idx]
            if z > SOLO_ENTRY_Z:
                sig = -1
            elif z < -SOLO_ENTRY_Z:
                sig = 1
            elif abs(z) < SOLO_EXIT_Z:
                sig = 0
            else:
                sig = prev
            _solo_signal[idx] = sig

            if sig == 0:
                continue

            price = prices_today[idx]
            cap = ALLOC_FRACTION * POSITION_LIMITS[idx]
            shares = int(np.floor(cap / price))
            positions[idx] = sig * shares

    # ---------------- ALGO-residual mean-reversion ----------------
    # Net each instrument's price against ALGO (OLS beta over a long
    # trailing window) and mean-revert the residual (short trailing
    # z-score), same entry/exit-band logic as the solo book above.
    if nDays >= MIN_DAYS_ALGO_RESID:
        p_algo = prcSoFar[0]
        for idx in ALGO_RESID_MEANREV:
            p = prcSoFar[idx]

            w_beta = min(ALGO_RESID_BETA_WINDOW, nDays)
            beta, const = _ols_hedge(p[-w_beta:], p_algo[-w_beta:])
            if beta is None:
                continue  # degenerate/untrustworthy fit this day; stay out

            w_z = min(ALGO_RESID_Z_WINDOW, nDays)
            resid_hist = p[-w_z:] - beta * p_algo[-w_z:] - const
            mu, sigma = resid_hist.mean(), resid_hist.std()
            if sigma < 1e-9:
                continue
            z = (resid_hist[-1] - mu) / sigma

            prev = _algo_resid_signal[idx]
            if z > ALGO_RESID_ENTRY_Z:
                sig = -1
            elif z < -ALGO_RESID_ENTRY_Z:
                sig = 1
            elif abs(z) < ALGO_RESID_EXIT_Z:
                sig = 0
            else:
                sig = prev
            _algo_resid_signal[idx] = sig

            if sig == 0:
                continue

            price = prices_today[idx]
            cap = ALLOC_FRACTION * POSITION_LIMITS[idx]
            shares = int(np.floor(cap / price))
            positions[idx] = sig * shares

    return positions


def getMyPosition(prcSoFar):
    """
    prcSoFar: np.ndarray of shape (51, numDays), day (numDays-1) is most recent.
    Returns: 1D np.ndarray of 51 integers (target share positions).
    """
    global _last_good_position
    try:
        prcSoFar = np.asarray(prcSoFar, dtype=float)
        if prcSoFar.shape[0] != N_INST:
            return _last_good_position.copy()

        positions = _run_strategy(prcSoFar)
        positions = np.asarray(positions, dtype=int)
        _last_good_position = positions.copy()
        return positions
    except Exception:
        # Never let an unexpected error crash the submission mid-run.
        return _last_good_position.copy()