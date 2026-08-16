"""
Drop-in replacement for ALGO-only residualization in the CS book.
Fit ONCE on your training window (e.g. days 1-750), then reuse the
fixed loadings live/OOS. Do not refit per-day unless you deliberately
want a rolling-refit version (slower, and re-introduces backtest risk
this close to deadline -- fixed loadings is the safer choice now).

Usage:
    resid_model = fit_pca_residualizer(train_asset_returns_df, n_components=3)
    residuals_df = resid_model.transform(full_asset_returns_df)

Then feed residuals_df into your existing 5-day rolling sum / rank /
3-long-3-short signal exactly as you did with the ALGO-only residuals.
"""
import numpy as np
import pandas as pd


class PCAResidualizer:
    def __init__(self, mu, sigma, eigvecs_k):
        self.mu = mu                # pd.Series, per-asset mean (train)
        self.sigma = sigma          # pd.Series, per-asset std (train)
        self.W = eigvecs_k          # np.ndarray (n_assets, k), fixed loadings

    def transform(self, asset_rets: pd.DataFrame) -> pd.DataFrame:
        """asset_rets: DataFrame [days x assets], same column order as fit."""
        cols = asset_rets.columns
        std = (asset_rets[cols] - self.mu[cols]) / self.sigma[cols]
        scores = std.values @ self.W
        recon_std = scores @ self.W.T
        resid_std = std.values - recon_std
        resid = resid_std * self.sigma[cols].values
        return pd.DataFrame(resid, index=asset_rets.index, columns=cols)


def fit_pca_residualizer(train_asset_rets: pd.DataFrame, n_components: int = 3) -> PCAResidualizer:
    """
    train_asset_rets: DataFrame [train_days x assets] of RAW asset returns
                       (exclude ALGO -- this replaces the ALGO regression
                       entirely, it doesn't need ALGO as an input at all).
    n_components: how many PCs to remove (3 validated OOS; 5 tested slightly
                  better in our proxy test -- verify 5 in your real scorer
                  before trusting it over 3).
    """
    mu = train_asset_rets.mean()
    sigma = train_asset_rets.std()
    std = (train_asset_rets - mu) / sigma

    cov = np.cov(std.values, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    W_k = eigvecs[:, :n_components]
    return PCAResidualizer(mu, sigma, W_k)


if __name__ == "__main__":
    # Smoke test against prices.csv to confirm this matches the validated numbers
    df = pd.read_csv('prices.csv')
    df.index = np.arange(1, len(df) + 1)
    rets = df.pct_change().dropna()
    assets = [c for c in rets.columns if c != 'ALGO']
    asset_rets = rets[assets]

    TRAIN_END = 750
    train = asset_rets[asset_rets.index <= TRAIN_END]

    for k in (3, 5):
        model = fit_pca_residualizer(train, n_components=k)
        resid = model.transform(asset_rets)
        print(f"k={k}: residual matrix shape {resid.shape}, "
              f"train-window resid std mean {resid.loc[resid.index <= TRAIN_END].std().mean():.5f}")