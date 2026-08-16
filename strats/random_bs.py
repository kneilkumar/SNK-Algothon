import numpy as np
import pandas as pd
import torch
import optuna
import optunahub

N_INST = 51
POSITION_LIMITS = torch.tensor([100_000.0] + [10_000.0] * 50)

PAIRS = [
    (10, 46),  # SMAH ~ ILVX   (EG p=3.4e-9)
    (1, 20),  # AENO ~ NWIG   (EG p=3.7e-9)
    (8, 27),  # HUXZ ~ ACAC   (EG p=1.2e-8)
    (7, 40),  # HETT ~ ULXY   (EG p=2.1e-7)
    (25, 37),  # CTGI ~ EELT   (EG p=7.9e-6, both legs individually stationary too)
    (18, 35),  # RTTH ~ NAYO   (EG p=3.5e-5)
    (41, 36),  # BLBT ~ FWWG
    (26, 32),  # ALUT ~ CCNS
    (13, 45),  # EORC ~ NGTE
    (33, 42),  # MTNS ~ BENI
]

mr_param_list = {"hedge_window": 300,
                    "z_window": 40,
                    "entry_z": 1.25,
                    "exit_z": 0.25,
                    "alloc_frac": 0.95,
                    "min_days_pairs": 40+2}


_pair_signal = {i: 0 for i in range(len(PAIRS))}


def read_data(price_data):
    prices = pd.read_csv(price_data)
    return (prices.values).T


def mean_reversion_strat(price_data, params, t, trial, optimiser=None):
    prcSoFar = price_data

    def _ols_hedge(a, b):
        """Closed-form OLS hedge ratio & intercept for a ~ hedge*b + const."""
        b_mean = torch.mean(b)
        a_mean = torch.mean(a)
        var = torch.dot(b - b_mean, b - b_mean)
        if var < 1e-9:
            return None, None
        hedge = torch.dot(b - b_mean, a - a_mean) / var
        const = a_mean - hedge * b_mean
        return hedge, const

    nInst, nDays = prcSoFar.shape
    positions = torch.zeros(nInst)
    prices_today = prcSoFar[:, -1]

    # ---------------- Pair trades ----------------
    if nDays >= params["min_days_pairs"]:
        for i, (idxA, idxB) in enumerate(PAIRS):
            pa = prcSoFar[idxA]
            pb = prcSoFar[idxB]

            w_hedge = min(params["hedge_window"], nDays)
            hedge, const = _ols_hedge(pa[-w_hedge:], pb[-w_hedge:])
            if hedge is None or hedge <= 0:
                continue  # degenerate/untrustworthy fit this day; stay out
            w_z = min(params["z_window"], nDays)
            spread_hist = pa[-w_z:] - hedge * pb[-w_z:] - const
            mu, sigma = torch.mean(spread_hist), torch.std(spread_hist)
            z = (spread_hist[-1] - mu) / sigma

            k_entry, k_exit = torch.tensor(5.,requires_grad=True),torch.tensor(5., requires_grad=True)
            entry_signal = -torch.tanh(k_entry * (z - params["entry_z"])) # handles long/short direction + saturation
            magnitude_gate = torch.sigmoid(k_exit * (torch.abs(z) - params['exit_z']))  # handles the "are we outside the deadzone" gate
            sig = entry_signal * magnitude_gate

            price_a, price_b = prices_today[idxA], prices_today[idxB]
            cap_a = params["alloc_frac"] * POSITION_LIMITS[idxA]
            cap_b = params["alloc_frac"] * POSITION_LIMITS[idxB]

            max_shares_a_from_a = cap_a / price_a
            max_shares_a_from_b = cap_b / (hedge * price_b)
            shares_a = torch.min(max_shares_a_from_a, max_shares_a_from_b)
            shares_b = shares_a * hedge

            positions[idxA] += sig * shares_a
            positions[idxB] += -sig * shares_b

    return positions, optimiser


def momentum_strat(price_data, params, t):
    pass


def backprop_optimisation(params, strategy, price_data, init):
    price_data = torch.tensor(price_data, dtype=torch.float32)
    params = {key: (torch.tensor(value, dtype=torch.int, requires_grad=False) if key in ("z_window", "hedge_window") else torch.tensor(value, requires_grad=True, dtype=torch.float32)) for key, value in params.items()}
    optimiser = torch.optim.RMSprop(params=list(params.values()), lr=0.0001)
    if init:
        numtestdays = 500
    else:
        numtestdays = 300 # fill in properly later

    strategy_fn = lambda prcHistSoFar: strategy(prcHistSoFar, params, prcHistSoFar.shape[1], trial=None)[0]
    plmu, plstd, pll = backtest_live(strategy_fn, nInst=N_INST, prcHist=price_data, numTestDays=numtestdays,
                                     optimiser=None)
    score = -1 * scoring_calculation(plmu, plstd)
    current_score = score.item()
    for i in range(75):
        optimiser.zero_grad()
        score.backward()
        optimiser.step()
        strategy_fn = lambda prcHistSoFar: strategy(prcHistSoFar, params, prcHistSoFar.shape[1], trial=None)[0]
        plmu, plstd, pll = backtest_live(strategy_fn, nInst=N_INST, prcHist=price_data, numTestDays=numtestdays,
                                         optimiser=None)
        new_score = -1 * scoring_calculation(plmu, plstd)
        print(new_score.item())
        score = new_score
    return score, strategy


def bayesian_optimisation(params, strategy, price_data, nDays, init):
    price_data = torch.tensor(price_data, dtype=torch.float32)
    if init:
        numtestdays = 500
    else:
        numtestdays = nDays

    def objective(trial):
        entry_z = trial.suggest_float("entry_z", 0.5, 3.0)
        exit_z = trial.suggest_float("exit_z", 0.0, entry_z)  # keep exit_z < entry_z structurally
        alloc_frac = trial.suggest_float("alloc_frac", 0.01, 1.0)
        hedge_window = trial.suggest_int("hedge_window", 50, 300)
        z_window = trial.suggest_int("z_window", 10, 100)


        params = {
            "entry_z": entry_z,
            "exit_z": exit_z,
            "alloc_frac": alloc_frac,
            "hedge_window": hedge_window,
            "z_window": z_window,
            "min_days_pairs": z_window + 2,
        }

        strategy_fn = lambda prcHistSoFar: strategy(prcHistSoFar, params, prcHistSoFar.shape[1], trial=None)[0]
        plmu_reopt, plstd_reopt, pll = backtest_live(strategy_fn, nInst=N_INST, prcHist=price_data, numTestDays=numtestdays,
                                         optimiser='bayes')

        return scoring_calculation(plmu_reopt, plstd_reopt)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=5)
    return study.best_params


def backtest_live(strategy, nInst, prcHist, numTestDays, optimiser, verbose=True):

    def chargeFees(dvolumes, commRate):
        """
        Total commission for one day's trades.
        """
        return torch.sum(dvolumes * commRate)

    """
    Function to loop over days and calculate/store PnLs
    """

    # initial values
    cash = torch.tensor(0, dtype=torch.float32)
    curPos = torch.zeros(nInst)

    commRate = torch.full_like(curPos, 0.0001)
    commRate[0] = torch.tensor(0.00002)

    dlrPosLimit = torch.full_like(curPos, 10_000)
    dlrPosLimit[0] = torch.tensor(100_000)

    totDVolume = torch.tensor(0, dtype=torch.float32)
    value = torch.tensor(0, dtype=torch.float32)
    comm = torch.tensor(0, dtype=torch.float32)

    todayPLL = []
    _, nt = prcHist.shape
    # start day is the first day to run getPosition() on
    # e.g. startDay=500 if last 250 of 750 days used as test days
    startDay = nt - numTestDays + 1

    for t in range(startDay, nt + 1):
        # price history up to and including t, e.g. if t=500, gets first 500 days
        prcHistSoFar = prcHist[:, :t]
        curPrices = torch.tensor(prcHistSoFar[:, -1])

        # trading loop, do not do it on the very last day of the test
        if t < nt:
            # get new positions
            newPosOrig = strategy(prcHistSoFar)

            # clip to position limits, and enforce integer shares
            posLimits = (dlrPosLimit / curPrices)
            newPos = torch.clamp(newPosOrig, -posLimits, posLimits)
        else:
            # the final day is only used as 'mark' of final PnL
            newPos = curPos

        # change in positions
        deltaPos = newPos - curPos

        cash -= curPrices.dot(deltaPos) + comm

        # calculate commissions
        dvolumes = curPrices * torch.abs(deltaPos)
        dvolume = torch.sum(dvolumes)
        totDVolume += dvolume
        comm = chargeFees(dvolumes, commRate)

        curPos = torch.tensor(newPos)
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

    pll = torch.stack(todayPLL)
    plmu, plstd = (torch.mean(pll), torch.std(pll))

    # calculate annualised Sharpe
    annSharpe = 0.0
    if plstd > 0:
        annSharpe = torch.sqrt(torch.tensor(250)) * plmu / plstd

    return plmu, plstd, pll


def scoring_calculation(mean_pl, std_pl):
    sr = torch.sqrt(torch.tensor(250)) * mean_pl / std_pl
    fraction = (sr**2) / (1 + (sr**2))
    return mean_pl * fraction


def reoptimisation_trigger(recent_window_size, pl_array, current_strat, alt_strat, price_data, current_params, init_opt):
    if init_opt:
        train_data = price_data[:, :500]
        init_bayes_params = bayesian_optimisation(current_params, current_strat, train_data,500, True)
        # init_bprop_params = backprop_optimisation(current_params, current_strat, train_data, True)
        # strategy_fn = lambda prcHistSoFar: current_strat(prcHistSoFar, init_bayes_params, prcHistSoFar.shape[1], trial=None)[0]
        # plmu_bayes, plstd_bayes, pll_bayes = backtest_live(strategy_fn, nInst=N_INST, prcHist=price_data, numTestDays=500,
        #                                  optimiser=None)
        # strategy_fn = lambda prcHistSoFar: current_strat(prcHistSoFar, init_bprop_params, prcHistSoFar.shape[1], trial=None)[0]
        # plmu_bprop, plstd_bprop, pll = backtest_live(strategy_fn, nInst=N_INST, prcHist=price_data, numTestDays=500,
        #                                  optimiser=None)
        # init_bayes_score = scoring_calculation(plmu_bayes, plstd_bayes)
        # init_bprop_score = scoring_calculation(plmu_bprop, plstd_bprop)

        # if init_bayes_score >= init_bprop_score:
        #     params = init_bayes_params
        #     return current_strat, params
        # else:
        #     params = init_bprop_params
        #     return current_strat, params

        return current_strat, init_bayes_params

    recent_window = torch.tensor(pl_array[-recent_window_size:])
    best_score = 0

    recent_window_train_len = int(np.floor(len(recent_window) * 0.6))
    recent_window_test_len = int(np.floor(len(recent_window) * 0.4))

    score_losing_streak = scoring_calculation(torch.mean(recent_window[recent_window_train_len:]), torch.std(recent_window[recent_window_train_len:]))  # recalc on final 40%, this is the test set
    best_params_bayes = bayesian_optimisation(current_params,current_strat, price_data, recent_window_train_len, False)  # price data needs to include first 60% of the recent window
    # best_params_bprop = backprop_optimisation(current_params, current_strat,price_data, recent_window_train_len, False)  # price data needs to include first 60% of the recent window
    strategy_fn = lambda prcHistSoFar: mean_reversion_strat(prcHistSoFar, best_params_bayes, prcHistSoFar.shape[1], trial=None)[0]
    plmu_bayes, plstd_bayes, pll_bayes = backtest_live(strategy_fn, nInst=N_INST, prcHist=price_data, numTestDays=len(recent_window),optimiser=None) # this re-eval needs to include last 40% of the recent window
    # strategy_fn = lambda prcHistSoFar: mean_reversion_strat(prcHistSoFar, best_params_bprop, prcHistSoFar.shape[1], trial=None)[0]
    # plmu_bprop, plstd_bprop, pll_bprop = backtest_live(strategy_fn, nInst=N_INST, prcHist=price_data, numTestDays=len(recent_window),optimiser=None)
    bayes_reoptimised_score = scoring_calculation(plmu_bayes, plstd_bayes)
    # bprop_reoptimised_score = scoring_calculation(plmu_bprop, plstd_bprop)

    if (bayes_reoptimised_score > score_losing_streak):
            new_params = best_params_bayes
            best_score = bayes_reoptimised_score
            return current_strat, new_params
    else:
        strat_temp_var = current_strat
        best_params_bayes = bayesian_optimisation(current_params, alt_strat, price_data) # price data needs to include first 60% of the recent window
        best_params_bprop = backprop_optimisation(current_params, alt_strat, price_data) # price data needs to include first 60% of the recent window
        strategy_fn = lambda prcHistSoFar: alt_strat(prcHistSoFar, best_params_bayes, prcHistSoFar.shape[1], trial=None)[0]
        plmu_bayes, plstd_bayes, pll_bayes = backtest_live(strategy_fn, nInst=N_INST, prcHist=price_data,
                                                           numTestDays=recent_window,
                                                           optimiser=None)  # this re-eval needs to include last 40% of the recent window
        strategy_fn = lambda prcHistSoFar: alt_strat(prcHistSoFar, best_params_bprop, prcHistSoFar.shape[1], trial=None)[0]
        plmu_bprop, plstd_bprop, pll_bprop = backtest_live(strategy_fn, nInst=N_INST, prcHist=price_data,
                                                           numTestDays=recent_window, optimiser=None)
        bayes_reoptimised_score = scoring_calculation(plmu_bayes, plstd_bayes)
        bprop_reoptimised_score = scoring_calculation(plmu_bprop, plstd_bprop)
        if (bayes_reoptimised_score > score_losing_streak) or (bprop_reoptimised_score > score_losing_streak):
            if bayes_reoptimised_score >= bprop_reoptimised_score:
                new_params = best_params_bayes
                current_strat = alt_strat
                alt_strat = strat_temp_var
            else:
                new_params = best_params_bprop
                current_strat = alt_strat
                alt_strat = strat_temp_var

    if (bayes_reoptimised_score <= score_losing_streak) and (bprop_reoptimised_score <= score_losing_streak):
        pass # do nothing or switch to third strat (atm) do nothing since its noisy)
    else:
        return current_strat, current_params, alt_strat


def live_re_eval_twin(numTestDays, current_strat, params, lookbackwindow):
    nInst = 51
    nt = 1000

    pricesFile = "./prices.txt"
    numTestDays = numTestDays

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
        Load prices from csv file (one instrument per column) and transpose into
        one instrument per row.
        """
        global nt, nInst

        df = pd.read_csv(fn, sep=r"\s+", header=0, index_col=None)
        nt, nInst = df.shape

        return torch.tensor(df.values.T, dtype=torch.float32)


    def chargeFees(dvolumes, commRate):
        """
        Total commission for one day's trades.
        """
        return torch.sum(dvolumes * commRate)


    def score(mu, sigma, param=scoreDefaultParam):
        """
        Final score from the daily-PnL mean & std.
        Fully differentiable.
        """
        sr = torch.sqrt(torch.tensor(250.0, device=mu.device)) * mu / (sigma + 1e-8)
        frac = sr**2 / (sr**2 + param**2)

        return torch.where(mu > 0, mu * frac, mu)


    def calcPL(prcHist, numTestDays, verbose=True):
        """
        Function to loop over days and calculate/store PnLs
        """

        reopted_params = None

        # initial values
        cash = torch.tensor(0.0)
        value = torch.tensor(0.0)
        comm = torch.tensor(0.0)
        totDVolume = torch.tensor(0.0)

        curPos = torch.zeros(nInst)

        todayPLL = []

        _, nt = prcHist.shape

        # start day is the first day to run getPosition()
        startDay = nt - numTestDays

        for t in range(startDay, nt + 1):

            prcHistSoFar = prcHist[:, :t]
            curPrices = prcHistSoFar[:, -1]

            if t < nt:
                if reopted_params is not None:
                    newPosOrig = current_strat(prcHistSoFar, reopted_params, 0,0)[0]
                else:
                    newPosOrig = current_strat(prcHistSoFar, params, 0, 0)[0]

                # clip to position limits
                posLimits = dlrPosLimit / curPrices
                newPos = torch.clamp(newPosOrig, -posLimits, posLimits)

            else:
                newPos = curPos

            deltaPos = newPos - curPos

            cash -= curPrices.dot(deltaPos) + comm

            dvolumes = curPrices * torch.abs(deltaPos)
            dvolume = torch.sum(dvolumes)

            totDVolume += dvolume

            comm = chargeFees(dvolumes, commRate)

            curPos = newPos

            posValue = curPos.dot(curPrices)

            todayPL = cash + posValue - value

            value = cash + posValue

            ret = torch.where(
                totDVolume > 0,
                value / (totDVolume + 1e-8),
                torch.tensor(0.0, device=value.device),
            )

            if t > startDay:

                if verbose:
                    print(
                        f"Day {t} value: {value.item():.2f} "
                        f"todayPL: ${todayPL.item():.2f} "
                        f"$-traded: {totDVolume.item():.0f} "
                        f"return: {ret.item():.5f}"
                    )

                todayPLL.append(todayPL)

                if torch.sum(torch.stack(todayPLL[-lookbackwindow:])) <= 100:
                    reopted_params = reoptimisation_trigger(lookbackwindow,todayPLL,current_strat,momentum_strat,prcAll, params,False,)

        pll = torch.stack(todayPLL)

        plmu = torch.mean(pll)
        plstd = torch.std(pll)

        annSharpe = torch.where(
            plstd > 0,
            torch.sqrt(torch.tensor(250.0, device=plmu.device)) * plmu / plstd,
            torch.tensor(0.0, device=plmu.device),
        )

        return (plmu, ret, plstd, annSharpe, totDVolume)


    prcAll = loadPrices(pricesFile)
    print(f"Loaded {nInst} instruments for {nt} days")

    # initialise commissions and position limits
    commRate = torch.full((nInst,), defaultCommRate, dtype=torch.float32)
    commRate[0] = inst0CommRate

    dlrPosLimit = torch.full((nInst,), defaultDlrPosLimit, dtype=torch.float32)
    dlrPosLimit[0] = inst0DlrPosLimit

    meanpl, ret, plstd, sharpe, dvol = calcPL(prcAll, numTestDays, verbose=True)

    scoreVal = score(meanpl, plstd, scoreDefaultParam)

    print("=====")
    print(f"mean(PL): {meanpl.item():.1f}")
    print(f"return: {ret.item():.5f}")
    print(f"StdDev(PL): {plstd.item():.2f}")
    print(f"annSharpe(PL): {sharpe.item():.2f}")
    print(f"totDvolume: {dvol.item():.0f}")
    print(f"Score: {scoreVal.item():.2f}")


def main():
    prices = read_data('prices.csv')
    best_strat, best_params = reoptimisation_trigger(0, [], mean_reversion_strat, momentum_strat, prices, mr_param_list, True)
    best_params['min_days_pairs'] = best_params['z_window'] + 2
    live_re_eval_twin(500, best_strat, best_params, 50)
    pass


if __name__ == "__main__":
    main()
