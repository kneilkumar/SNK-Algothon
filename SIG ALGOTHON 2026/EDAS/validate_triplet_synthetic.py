import time
import numpy as np
from johansen_engine import johansen_engine

rng = np.random.default_rng(11)
n_inst = 15
n_t = 700

market_factor = np.cumsum(rng.normal(0.02, 1.0, n_t))
prices = np.zeros((n_inst, n_t))
for i in range(n_inst):
    idio = np.cumsum(rng.normal(0, 1.0, n_t))
    beta_mkt = rng.uniform(0.3, 1.2)
    prices[i] = 100 + beta_mkt * market_factor + idio

# Engineer ONE genuine triplet: inst_2 is a linear combo of inst_0 and
# inst_1 plus a stationary AR(1) residual. inst_0 and inst_1 themselves are
# just two more idiosyncratic random walks (not pairwise cointegrated with
# each other or with inst_2 alone) -- only the 3-way combination is
# stationary, so this should show up at group_size=3 and NOT at group_size=2.
h0, h1 = 0.8, 1.3
stat_noise = np.zeros(n_t)
phi_true = 0.8
eps = rng.normal(0, 1.0, n_t)
for t in range(1, n_t):
    stat_noise[t] = phi_true * stat_noise[t - 1] + eps[t]
prices[2] = h0 * prices[0] + h1 * prices[1] + stat_noise

names = [f"inst_{i}" for i in range(n_inst)]

t0 = time.time()
out = johansen_engine(
    prices,
    instrument_names=names,
    group_size=3,
    train_frac=0.67,
    fwer_alpha=0.05,
    min_cycles=3.0,
    n_null_sims=2000,
    seed=2,
    verbose=True,
)
print(f"elapsed: {time.time()-t0:.1f}s")

print("\n--- final candidates (group_size=3) ---")
cols = ["combo", "trace_stat", "p_value_sim", "pass_bonferroni",
        "half_life_train", "holdout_trace_stat", "holdout_half_life",
        "final_candidate", "coint_vector_normalized"]
print(out["candidates"][cols] if len(out["candidates"]) else "none")

# sanity: also run group_size=2 on the same data, expect inst_0/inst_1/inst_2
# pairs NOT to show up as Bonferroni survivors (the cointegration is only a
# 3-way relationship)
out2 = johansen_engine(
    prices, instrument_names=names, group_size=2, train_frac=0.67,
    fwer_alpha=0.05, min_cycles=3.0, n_null_sims=2000, seed=2, verbose=False,
)
print("\ngroup_size=2 candidates on the same data (expect none/unrelated):")
print(out2["candidates"][["combo", "trace_stat", "pass_bonferroni"]] if len(out2["candidates"]) else "none")
