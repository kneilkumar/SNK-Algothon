import pandas as pd
import numpy as np
from scipy.stats import linregress
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import adfuller

# ===========================
# Step 1: Load and normalize returns
# ===========================

prices = pd.read_csv("prices.csv")
prices = prices.iloc[:350]

returns = prices.pct_change().dropna()
returns = returns.iloc[-252:]

mean_returns = returns.mean(axis=0)
std_returns = returns.std(axis=0)

returns_normalised = (returns - mean_returns) / std_returns

# ===========================
# Step 2: PCA
# ===========================

pca = PCA(n_components=12)
eigenportfolio_returns = pca.fit_transform(returns_normalised)

print("Explained variance ratio:")
print(pca.explained_variance_ratio_)

X = np.column_stack([
    np.ones(len(eigenportfolio_returns)),
    eigenportfolio_returns
])

residuals = pd.DataFrame(index=returns.index, columns=returns.columns)
betas = {}

for instrument in returns.columns:
    y = returns[instrument].values

    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    fitted = X @ coef

    residuals[instrument] = y - fitted
    betas[instrument] = coef

# Convert residual returns into residual price process
X = residuals.cumsum()

# ===========================
# OU Estimation
# ===========================

def fit_OU_proc(data, dt=1):

    lagged = data[:-1]
    current = data[1:]

    phi, intercept, _, _, _ = linregress(lagged, current)

    if not (0 < phi < 1):
        return None

    errors = current - (phi * lagged + intercept)
    var_error = np.var(errors, ddof=2)

    theta = -np.log(phi) / dt
    mu = intercept / (1 - phi)
    sigma = np.sqrt(2 * theta * var_error / (1 - phi**2))
    half_life = np.log(2) / theta

    return {
        "phi": phi,
        "theta": theta,
        "mu": mu,
        "sigma": sigma,
        "half_life": half_life
    }

# ===========================
# Screening Parameters
# ===========================

ADF_THRESHOLD = 0.05
MIN_HALF_LIFE = 2
MAX_HALF_LIFE = 30

# ===========================
# Screening
# ===========================

screened = {}

for instrument in X.columns:

    series = X[instrument].dropna().values

    # ---------- ADF Test ----------
    adf_stat, adf_pval, *_ = adfuller(
        series,
        regression="c",
        autolag="BIC"
    )

    if adf_pval > ADF_THRESHOLD:
        continue

    # ---------- OU Fit ----------
    params = fit_OU_proc(series)

    if params is None:
        continue

    # ---------- Half-life Filter ----------
    if not (MIN_HALF_LIFE <= params["half_life"] <= MAX_HALF_LIFE):
        continue

    screened[instrument] = {
        "adf_pval": adf_pval,
        **params
    }

# ===========================
# Results
# ===========================

screened = (
    pd.DataFrame(screened)
    .T
    .sort_values("adf_pval")
)

print(screened)
print(f"\nSelected {len(screened)} instruments out of {len(returns.columns)}")



# ===========================
# Step 6: Calculate OU s-score
# ===========================

entry_threshold = 1.25
exit_threshold = 0.75

signals = {}

for instrument, params in screened.iterrows():

    series = residuals[instrument].dropna()

    theta = params["theta"]
    mu = params["mu"]
    sigma = params["sigma"]

    # equilibrium standard deviation
    sigma_eq = sigma / np.sqrt(2 * theta)

    # current OU score
    s_score = (series - mu) / sigma_eq

    current_score = s_score.iloc[-1]

    if current_score > entry_threshold:
        signal = "SHORT"

    elif current_score < -entry_threshold:
        signal = "LONG"

    elif abs(current_score) < exit_threshold:
        signal = "EXIT"

    else:
        signal = "HOLD"


    signals[instrument] = {
        "s_score": current_score,
        "signal": signal,
        "theta": theta,
        "half_life": params["half_life"],
        "mu": mu,
        "sigma_eq": sigma_eq
    }


signals = (
    pd.DataFrame(signals)
    .T
    .sort_values("s_score")
)

print(signals)
