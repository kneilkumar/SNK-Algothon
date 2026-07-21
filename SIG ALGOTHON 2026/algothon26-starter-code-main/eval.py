#!/usr/bin/env python
"""Algothon 2026 evaluation script.

Participants: write getMyPosition(prcSoFar) in SNK.py and update the imports below

Usage:
    python eval.py                                   # single run, original behaviour
    python eval.py --optimize grid                   # exhaustive grid search over PARAM_GRID
    python eval.py --optimize grid --params ENTRY_Z EXIT_Z   # grid search over just these
    python eval.py --optimize random --n-samples 500  # random search
"""

import argparse
import itertools
import random

import numpy as np
import pandas as pd

import SNK  # import the module itself so we can read/write its tunable parameters
from SNK import getMyPosition as getPosition

nInst = 0
nt = 0

pricesFile = "./prices.txt"
numTestDays = 250

# parameter for scoring function
scoreDefaultParam = 1.0

# commission rates (0.0001 = 1bp)
# SPECIAL rate for instrument 0
defaultCommRate = 0.0001
inst0CommRate = 0.00002

# position limits (dollars)
# SPECIAL position limit for instrument 0
defaultDlrPosLimit = 10_000
inst0DlrPosLimit = 100_000


def loadPrices(fn):
    """
    Load prices from csv file (one instrument per column) and transpose into one instrument per row
    """
    global nt, nInst
    df = pd.read_csv(fn, sep=r"\s+", header=0, index_col=None)
    nt, nInst = df.shape
    return (df.values).T


def chargeFees(dvolumes, commRate):
    """
    Total commission for one day's trades.
    """
    return np.sum(dvolumes * commRate)


def score(mu, sigma, param=scoreDefaultParam):
    """
    Final score from the daily-PnL mean & std, plus a scoring parameter.
    """
    if mu <= 0 or sigma < 1e-10:
        return mu
    sr = np.sqrt(250) * mu / sigma
    frac = sr**2 / (sr**2 + param**2)
    return mu * frac


def calcPL(prcHist, numTestDays, verbose=True):
    """
    Function to loop over days and calculate/store PnLs
    """

    # initial values
    cash = 0
    curPos = np.zeros(nInst)
    totDVolume = 0
    value = 0
    comm = 0

    todayPLL = []
    _, nt = prcHist.shape
    # start day is the first day to run getPosition() on
    # e.g. startDay=500 if last 250 of 750 days used as test days
    startDay = nt - numTestDays

    for t in range(startDay, nt + 1):
        # price history up to and including t, e.g. if t=500, gets first 500 days
        prcHistSoFar = prcHist[:, :t]
        curPrices = prcHistSoFar[:, -1]

        # trading loop, do not do it on the very last day of the test
        if t < nt:
            # get new positions
            newPosOrig = getPosition(prcHistSoFar)

            # clip to position limits, and enforce integer shares
            posLimits = (dlrPosLimit / curPrices).astype(int)
            newPos = np.clip(newPosOrig, -posLimits, posLimits).astype(int)
        else:
            # the final day is only used as 'mark' of final PnL
            newPos = np.array(curPos)

        # change in positions
        deltaPos = newPos - curPos

        cash -= curPrices.dot(deltaPos) + comm

        # calculate commissions
        dvolumes = curPrices * np.abs(deltaPos)
        dvolume = np.sum(dvolumes)
        totDVolume += dvolume
        comm = chargeFees(dvolumes, commRate)

        curPos = np.array(newPos)
        posValue = curPos.dot(curPrices)
        # PnL is the daily change in portfolio value (cash plus positions)
        todayPL = cash + posValue - value

        value = cash + posValue

        # calculate return (portfolio value over total dollar volume)
        ret = 0.0
        if totDVolume > 0:
            ret = value / totDVolume

        # only score for test days
        if t > startDay:
            if verbose:
                print(
                    f"Day {t} value: {value:.2f} todayPL: ${todayPL:.2f} $-traded: {totDVolume:.0f} return: {ret:.5f}"
                )
            todayPLL.append(todayPL)

    pll = np.array(todayPLL)
    plmu, plstd = (np.mean(pll), np.std(pll))

    # calculate annualised Sharpe
    annSharpe = 0.0
    if plstd > 0:
        annSharpe = np.sqrt(250) * plmu / plstd

    return (plmu, ret, plstd, annSharpe, totDVolume)


# ---------------------------------------------------------------------------
# Parameter-optimisation helpers.
#
# SNK.getMyPosition keeps its tunable knobs (ENTRY_Z, EXIT_Z, etc.) as
# module-level globals on SNK, and keeps trading state (which pair/solo
# position it currently holds) in a few more module-level globals
# (_pair_signal, _solo_signal, _last_good_position) that persist across
# days *within one run*. To backtest a different parameter set we (a) set
# the parameter globals to the values we want to try, and (b) reset the
# state globals back to flat before replaying the whole history - otherwise
# a later run would inherit whatever position the previous run's backtest
# ended up holding.
# ---------------------------------------------------------------------------

def reset_strategy_state():
    """Zero out any persistent module-level state SNK.py keeps between days,
    so consecutive backtests during a parameter sweep don't leak state
    from one run into the next."""
    for name in ("_pair_signal", "_solo_signal"):
        attr = getattr(SNK, name, None)
        if isinstance(attr, dict):
            setattr(SNK, name, {k: 0 for k in attr})
    attr = getattr(SNK, "_last_good_position", None)
    if isinstance(attr, np.ndarray):
        setattr(SNK, "_last_good_position", np.zeros_like(attr))


def apply_params(params):
    """Set each param as a module-level attribute on SNK, e.g. SNK.ENTRY_Z = 1.75.
    Silently skips any name SNK.py doesn't actually define, so this works
    no matter which subset of parameters your strategy exposes - it just
    won't do anything for names that don't exist."""
    applied = {}
    for k, v in params.items():
        if hasattr(SNK, k):
            setattr(SNK, k, v)
            applied[k] = v
        else:
            print(f"  (warning: SNK.py has no attribute '{k}', skipping)")
    return applied


def run_backtest(prcAll, numTestDays, params=None, verbose=False):
    """Run one full backtest with the given parameter overrides applied,
    starting from a clean (flat) strategy state, and return its score + stats."""
    applied = apply_params(params) if params else {}
    reset_strategy_state()
    meanpl, ret, plstd, sharpe, dvol = calcPL(prcAll, numTestDays, verbose=verbose)
    scoreVal = score(meanpl, plstd, scoreDefaultParam)
    return {
        "params": applied,
        "meanpl": meanpl,
        "ret": ret,
        "plstd": plstd,
        "sharpe": sharpe,
        "dvol": dvol,
        "score": scoreVal,
    }


# Candidate values to search over. Edit freely - add/remove parameter names
# and candidate values as needed. Any name here that isn't actually defined
# in SNK.py is skipped automatically (see apply_params above).
PARAM_GRID = {
    "ENTRY_Z": [1.25, 1.5, 1.75, 2.0, 2.5],
    "EXIT_Z": [0.25, 0.5, 0.75],
    "SOLO_ENTRY_Z": [1.0, 1.5, 2.0],
    "SOLO_EXIT_Z": [0.25, 0.5],
    "ALLOC_FRACTION": [0.8, 0.9, 0.95, 1.0],
    "HEDGE_WINDOW": [150, 300, 450],
    "Z_WINDOW": [20, 40, 60],
    "SOLO_WINDOW": [40, 60, 80],
}


def grid_search(prcAll, numTestDays, grid=PARAM_GRID, keys=None):
    """Exhaustive search over every combination of the given parameters.

    WARNING: combinations multiply fast. The full PARAM_GRID above is
    5*3*3*2*4*3*3*3 = 14,580 combinations - each is a full 250-day backtest,
    so budget real time for it (a rough estimate: run one backtest by hand
    and multiply). Pass `--params` to restrict to a handful of names if you
    just want to sweep ENTRY_Z/EXIT_Z, or use random_search for a big grid.
    """
    if keys is None:
        keys = list(grid.keys())
    missing = [k for k in keys if not hasattr(SNK, k)]
    if missing:
        print(f"(note: SNK.py doesn't define {missing}, they'll be ignored)")

    combos = list(itertools.product(*(grid[k] for k in keys)))
    print(f"Grid search: {len(combos)} combinations over {keys}")

    results = []
    report_every = max(1, len(combos) // 20)
    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        results.append(run_backtest(prcAll, numTestDays, params, verbose=False))
        if (i + 1) % report_every == 0 or i == len(combos) - 1:
            best = max(results, key=lambda r: r["score"])
            print(f"  [{i + 1}/{len(combos)}] best score so far: {best['score']:.2f}")
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def random_search(prcAll, numTestDays, grid=PARAM_GRID, keys=None, n_samples=200, seed=0):
    """Randomly sample n_samples parameter combinations - use this instead of
    grid_search when the full grid is too large to search exhaustively."""
    if keys is None:
        keys = list(grid.keys())
    rng = random.Random(seed)
    print(f"Random search: {n_samples} samples over {keys}")

    results = []
    report_every = max(1, n_samples // 20)
    for i in range(n_samples):
        params = {k: rng.choice(grid[k]) for k in keys}
        results.append(run_backtest(prcAll, numTestDays, params, verbose=False))
        if (i + 1) % report_every == 0 or i == n_samples - 1:
            best = max(results, key=lambda r: r["score"])
            print(f"  [{i + 1}/{n_samples}] best score so far: {best['score']:.2f}")
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def print_top(results, n=10):
    print(f"\nTop {min(n, len(results))} parameter sets by score:")
    for r in results[:n]:
        print(
            f"  score={r['score']:.2f}  sharpe={r['sharpe']:.2f}  "
            f"meanPL={r['meanpl']:.1f}  stdPL={r['plstd']:.1f}  params={r['params']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the algothon evaluation, optionally sweeping strategy parameters."
    )
    parser.add_argument(
        "--optimize", choices=["grid"], default=None,
        help="Sweep parameters instead of running a single backtest with SNK.py's current values.",
    )
    parser.add_argument(
        "--params", nargs="*", default=None,
        help="Restrict the sweep to these parameter names (default: everything in PARAM_GRID).",
    )
    parser.add_argument(
        "--n-samples", type=int, default=200,
        help="Number of samples to draw for --optimize random (default 200).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for --optimize random.")
    parser.add_argument("--top", type=int, default=10, help="How many top results to print.")
    args = parser.parse_args()

    prcAll = loadPrices(pricesFile)
    print(f"Loaded {nInst} instruments for {nt} days")

    # initialise the per-instrument commissions and position limits
    commRate = np.full(nInst, defaultCommRate)
    commRate[0] = inst0CommRate
    dlrPosLimit = np.full(nInst, defaultDlrPosLimit)
    dlrPosLimit[0] = inst0DlrPosLimit

    if args.optimize == "grid":
        results = grid_search(prcAll, numTestDays, keys=args.params)
        print_top(results, args.top)
    elif args.optimize == "random":
        results = random_search(prcAll, numTestDays, keys=args.params, n_samples=args.n_samples, seed=args.seed)
        print_top(results, args.top)
    else:
        meanpl, ret, plstd, sharpe, dvol = calcPL(prcAll, numTestDays, verbose=True)
        scoreVal = score(meanpl, plstd, scoreDefaultParam)
        print("=====")
        print(f"mean(PL): {meanpl:.1f}")
        print(f"return: {ret:.5f}")
        print(f"StdDev(PL): {plstd:.2f}")
        print(f"annSharpe(PL): {sharpe:.2f}")
        print(f"totDvolume: {dvol:.0f}")
        print(f"Score: {scoreVal:.2f}")